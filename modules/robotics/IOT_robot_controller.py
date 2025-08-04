import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import time
import platform

from modules.decision import ExecutiveDecisionNet

class ParietalCortexEmulator(nn.Module):
    def __init__(self, tactile_dim: int, vision_dim: int, audio_dim: int, intent_dim: int, output_dim: int):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(tactile_dim + vision_dim + audio_dim + intent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, tactile: torch.Tensor, vision: torch.Tensor, audio: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([tactile, vision, audio, intent], dim=-1)
        return self.fusion_layer(combined)


class RoboticsController(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.executive_function = ExecutiveDecisionNet(input_channels=3, output_dim=512).to(device)
        self.parietal_module = ParietalCortexEmulator(
            tactile_dim=64, vision_dim=512, audio_dim=64, intent_dim=64, output_dim=128
        )
        self.motor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )

    def forward(self,
                oni_vision_features: torch.Tensor,
                oni_audio_features: torch.Tensor,
                oni_tactile_features: torch.Tensor,
                intent_vector: torch.Tensor) -> Dict:
        parietal_out = self.parietal_module(
            tactile=oni_tactile_features,
            vision=oni_vision_features,
            audio=oni_audio_features,
            intent=intent_vector
        )
        motor_command = self.motor_head(parietal_out)
        return {
            'parietal_output': parietal_out,
            'motor_command': motor_command
        }


def issue_motor_command(motor_command: torch.Tensor, mode="pi"):
    motor_command = motor_command.squeeze().detach().cpu().numpy()

    if mode == "pi":
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin, val in enumerate(motor_command):
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH if val > 0 else GPIO.LOW)

    elif mode == "arduino":
        import serial
        arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        cmd = ",".join([str(int(val > 0)) for val in motor_command]) + "\n"
        arduino.write(cmd.encode('utf-8'))

    elif mode == "esp32":
        import serial
        esp = serial.Serial('/dev/ttyUSB1', 115200, timeout=1)
        cmd = ",".join([str(int(val > 0)) for val in motor_command]) + "\n"
        esp.write(cmd.encode('utf-8'))

    elif mode == "motor_controller":
        import smbus
        bus = smbus.SMBus(1)
        address = 0x40  # example I2C address
        data = [int(val * 255) for val in motor_command]
        bus.write_i2c_block_data(address, 0x00, data)

    else:
        print("[WARN] Unknown mode. No motor command issued.")


def oni_controller_runtime_loop(controller: RoboticsController, oni, output_mode="pi"):
    try:
        while True:
            oni_vision = oni.vision.get_features()
            oni_audio = oni.audio.get_features()
            oni_tactile = oni.haptics.get_features()
            oni_intent = oni.mind.get_intent_vector()

            output = controller(
                oni_vision_features=oni_vision,
                oni_audio_features=oni_audio,
                oni_tactile_features=oni_tactile,
                intent_vector=oni_intent
            )

            issue_motor_command(output['motor_command'], mode=output_mode)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("ONI RoboticsController shutdown.")
        if output_mode == "pi":
            import RPi.GPIO as GPIO
            GPIO.cleanup()
