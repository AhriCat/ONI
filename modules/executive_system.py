"""
Advanced Executive Function System - Vertical Column Model
========================================================

A modularized, optimized executive function system with relative imports and clear module boundaries.

Modules included:
- config.py
- models.py
- sensory.py
- motor.py
- memory.py
- executive.py
- vertical_column.py

"""

import asyncio
import logging
import time
from pathlib import Path
import torch

from .config import ExecutiveFunctionConfig
from .sensory import SensoryInput
from decision import TimeAwareExecutiveNet
from .motor import MotorOutput
from .memory import WorkingMemoryModule

logger = logging.getLogger(__name__)


class ExecutiveFunctionSystem:
    def __init__(self, config: ExecutiveFunctionConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.executive_net = TimeAwareExecutiveNet(config).to(self.device)

        self.is_running = False

        logger.info(f"System initialized on {self.device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.executive_net.parameters()):,}")

    async def process_input(self, sensory_input: SensoryInput):
        start_time = time.time()

        sensory_input = sensory_input.to(self.device)

        with torch.no_grad():
            output_logits, motor_output, monitoring_info = self.executive_net(sensory_input)

        processing_time = time.time() - start_time

        monitoring_info['processing_time'] = processing_time

        logger.info(f"Processed input in {processing_time:.4f}s")

        return output_logits, motor_output, monitoring_info

    async def start_processing_loop(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        self.is_running = True
        logger.info("Starting processing loop")

        while self.is_running:
            try:
                sensory_input = await asyncio.wait_for(input_queue.get(), timeout=1.0)
                result = await self.process_input(sensory_input)
                await output_queue.put(result)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(0.1)

    def stop_processing(self):
        self.is_running = False
        logger.info("Stopping processing loop")

    def save_checkpoint(self, step_count: int):
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_{step_count}.pt"
        torch.save({
            'model_state': self.executive_net.state_dict(),
            'config': self.config,
            'step': step_count
        }, checkpoint_path)

        logger.info(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.executive_net.load_state_dict(checkpoint['model_state'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
