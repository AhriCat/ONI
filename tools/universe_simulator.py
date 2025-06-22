import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.integrate as integrate
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import time
import threading
import queue
import json
import os
from pathlib import Path
import h5py
import astropy.units as u
from astropy.constants import G as gravitational_constant
import astropy.cosmology as cosmo
from astropy.cosmology import WMAP9 as default_cosmology
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CelestialBody:
    """Represents a celestial body in the universe simulation."""
    
    def __init__(self, name: str, mass: float, radius: float, position: np.ndarray, 
                 velocity: np.ndarray, color: str = 'blue', density: float = None):
        """
        Initialize a celestial body.
        
        Args:
            name: Name of the celestial body
            mass: Mass in kg
            radius: Radius in meters
            position: Initial position as [x, y, z] in meters
            velocity: Initial velocity as [vx, vy, vz] in m/s
            color: Color for visualization
            density: Density in kg/m³ (optional, calculated from mass and radius if not provided)
        """
        self.name = name
        self.mass = mass
        self.radius = radius
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color
        
        # Calculate density if not provided
        if density is None:
            volume = (4/3) * np.pi * (radius ** 3)
            self.density = mass / volume
        else:
            self.density = density
            
        # Initialize other properties
        self.acceleration = np.zeros(3)
        self.trajectory = [self.position.copy()]
        self.spin_axis = np.array([0, 0, 1])  # Default spin axis (north pole)
        self.spin_rate = 0.0  # Radians per second
        self.obliquity = 0.0  # Axial tilt in radians
        self.orbital_elements = {}
        
    def update_position(self, dt: float):
        """Update position based on velocity and time step."""
        self.position += self.velocity * dt
        self.trajectory.append(self.position.copy())
        
    def update_velocity(self, dt: float):
        """Update velocity based on acceleration and time step."""
        self.velocity += self.acceleration * dt
        
    def calculate_gravitational_force(self, other_body: 'CelestialBody') -> np.ndarray:
        """Calculate gravitational force between this body and another."""
        G = gravitational_constant.value  # Gravitational constant in m³/(kg·s²)
        
        # Vector from this body to the other body
        r_vec = other_body.position - self.position
        
        # Distance between bodies
        r = np.linalg.norm(r_vec)
        
        # Avoid division by zero or very small values
        if r < (self.radius + other_body.radius):
            # Bodies are overlapping, handle collision or use a minimum distance
            r = self.radius + other_body.radius
            
        # Calculate gravitational force magnitude
        force_magnitude = G * self.mass * other_body.mass / (r ** 2)
        
        # Calculate force vector (direction from this body to the other)
        force_vector = force_magnitude * r_vec / r
        
        return force_vector
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert celestial body to dictionary for serialization."""
        return {
            'name': self.name,
            'mass': self.mass,
            'radius': self.radius,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'color': self.color,
            'density': self.density,
            'spin_axis': self.spin_axis.tolist(),
            'spin_rate': self.spin_rate,
            'obliquity': self.obliquity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CelestialBody':
        """Create celestial body from dictionary."""
        body = cls(
            name=data['name'],
            mass=data['mass'],
            radius=data['radius'],
            position=data['position'],
            velocity=data['velocity'],
            color=data.get('color', 'blue'),
            density=data.get('density')
        )
        
        if 'spin_axis' in data:
            body.spin_axis = np.array(data['spin_axis'])
        if 'spin_rate' in data:
            body.spin_rate = data['spin_rate']
        if 'obliquity' in data:
            body.obliquity = data['obliquity']
            
        return body

class UniverseSimulator:
    """Simulator for celestial bodies in a universe."""
    
    def __init__(self, time_step: float = 3600.0, collision_detection: bool = True):
        """
        Initialize the universe simulator.
        
        Args:
            time_step: Simulation time step in seconds (default: 1 hour)
            collision_detection: Whether to detect and handle collisions
        """
        self.bodies = []
        self.time_step = time_step
        self.collision_detection = collision_detection
        self.time = 0.0  # Current simulation time in seconds
        self.history = []  # History of simulation states
        self.running = False
        self.simulation_thread = None
        self.event_queue = queue.Queue()
        self.cosmology = default_cosmology
        
    def add_body(self, body: CelestialBody) -> None:
        """Add a celestial body to the simulation."""
        self.bodies.append(body)
        logger.info(f"Added body: {body.name}")
        
    def remove_body(self, body_name: str) -> bool:
        """Remove a celestial body from the simulation by name."""
        for i, body in enumerate(self.bodies):
            if body.name == body_name:
                self.bodies.pop(i)
                logger.info(f"Removed body: {body_name}")
                return True
        logger.warning(f"Body not found: {body_name}")
        return False
    
    def update(self) -> None:
        """Update the simulation by one time step."""
        # Calculate gravitational forces and accelerations
        for i, body in enumerate(self.bodies):
            # Reset acceleration
            body.acceleration = np.zeros(3)
            
            # Calculate gravitational forces from all other bodies
            for j, other_body in enumerate(self.bodies):
                if i != j:  # Skip self-interaction
                    force = body.calculate_gravitational_force(other_body)
                    body.acceleration += force / body.mass
        
        # Update velocities based on accelerations
        for body in self.bodies:
            body.update_velocity(self.time_step)
        
        # Check for collisions if enabled
        if self.collision_detection:
            self._handle_collisions()
        
        # Update positions based on velocities
        for body in self.bodies:
            body.update_position(self.time_step)
        
        # Update simulation time
        self.time += self.time_step
        
        # Save current state to history (optional, can be memory intensive)
        if len(self.history) < 1000:  # Limit history to prevent memory issues
            self.history.append(self._get_current_state())
    
    def _handle_collisions(self) -> None:
        """Detect and handle collisions between bodies."""
        collisions = []
        
        # Detect collisions
        for i, body1 in enumerate(self.bodies[:-1]):
            for j, body2 in enumerate(self.bodies[i+1:], i+1):
                distance = np.linalg.norm(body1.position - body2.position)
                if distance < (body1.radius + body2.radius):
                    collisions.append((i, j))
        
        # Handle collisions (merge bodies)
        for i, j in reversed(collisions):  # Process in reverse to avoid index issues
            self._merge_bodies(i, j)
    
    def _merge_bodies(self, i: int, j: int) -> None:
        """Merge two colliding bodies."""
        body1 = self.bodies[i]
        body2 = self.bodies[j]
        
        # Calculate new properties
        total_mass = body1.mass + body2.mass
        
        # Conservation of momentum
        new_velocity = (body1.mass * body1.velocity + body2.mass * body2.velocity) / total_mass
        
        # New position at center of mass
        new_position = (body1.mass * body1.position + body2.mass * body2.position) / total_mass
        
        # New radius (assuming constant density)
        new_volume = (4/3) * np.pi * (body1.radius ** 3) + (4/3) * np.pi * (body2.radius ** 3)
        new_radius = ((3 * new_volume) / (4 * np.pi)) ** (1/3)
        
        # Create new body
        new_name = f"{body1.name}-{body2.name}"
        new_body = CelestialBody(
            name=new_name,
            mass=total_mass,
            radius=new_radius,
            position=new_position,
            velocity=new_velocity,
            color=body1.color  # Use color of larger body
        )
        
        # Replace the first body with the new one and remove the second
        self.bodies[i] = new_body
        self.bodies.pop(j)
        
        logger.info(f"Collision: {body1.name} and {body2.name} merged into {new_name}")
        
        # Add collision event to queue
        self.event_queue.put({
            'type': 'collision',
            'time': self.time,
            'bodies': [body1.name, body2.name],
            'result': new_name
        })
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the simulation."""
        return {
            'time': self.time,
            'bodies': [body.to_dict() for body in self.bodies]
        }
    
    def run_simulation(self, steps: int, callback=None) -> None:
        """
        Run the simulation for a specified number of steps.
        
        Args:
            steps: Number of time steps to simulate
            callback: Optional callback function called after each step
        """
        self.running = True
        
        for _ in range(steps):
            if not self.running:
                break
                
            self.update()
            
            if callback:
                callback(self._get_current_state())
        
        self.running = False
    
    def start_simulation_thread(self, steps: int, callback=None) -> None:
        """
        Start the simulation in a separate thread.
        
        Args:
            steps: Number of time steps to simulate
            callback: Optional callback function called after each step
        """
        if self.simulation_thread and self.simulation_thread.is_alive():
            logger.warning("Simulation already running")
            return
            
        self.simulation_thread = threading.Thread(
            target=self.run_simulation,
            args=(steps, callback)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info("Simulation thread started")
    
    def stop_simulation(self) -> None:
        """Stop the running simulation."""
        self.running = False
        
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)
            logger.info("Simulation stopped")
    
    def save_simulation(self, file_path: str) -> None:
        """
        Save the current simulation state to a file.
        
        Args:
            file_path: Path to save the simulation
        """
        data = {
            'time': self.time,
            'time_step': self.time_step,
            'collision_detection': self.collision_detection,
            'bodies': [body.to_dict() for body in self.bodies]
        }
        
        # Determine file format based on extension
        if file_path.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            with h5py.File(file_path, 'w') as f:
                # Store simulation parameters
                f.attrs['time'] = self.time
                f.attrs['time_step'] = self.time_step
                f.attrs['collision_detection'] = self.collision_detection
                
                # Create a group for bodies
                bodies_group = f.create_group('bodies')
                
                # Store each body
                for i, body in enumerate(self.bodies):
                    body_group = bodies_group.create_group(f'body_{i}')
                    body_group.attrs['name'] = body.name
                    body_group.attrs['mass'] = body.mass
                    body_group.attrs['radius'] = body.radius
                    body_group.attrs['color'] = body.color
                    body_group.attrs['density'] = body.density
                    body_group.attrs['spin_rate'] = body.spin_rate
                    body_group.attrs['obliquity'] = body.obliquity
                    
                    # Store arrays
                    body_group.create_dataset('position', data=body.position)
                    body_group.create_dataset('velocity', data=body.velocity)
                    body_group.create_dataset('spin_axis', data=body.spin_axis)
                    
                    # Store trajectory if available
                    if body.trajectory:
                        body_group.create_dataset('trajectory', data=np.array(body.trajectory))
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        logger.info(f"Simulation saved to {file_path}")
    
    def load_simulation(self, file_path: str) -> None:
        """
        Load a simulation state from a file.
        
        Args:
            file_path: Path to the simulation file
        """
        # Clear current simulation
        self.bodies = []
        
        # Determine file format based on extension
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            self.time = data['time']
            self.time_step = data['time_step']
            self.collision_detection = data['collision_detection']
            
            # Load bodies
            for body_data in data['bodies']:
                body = CelestialBody.from_dict(body_data)
                self.bodies.append(body)
                
        elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            with h5py.File(file_path, 'r') as f:
                # Load simulation parameters
                self.time = f.attrs['time']
                self.time_step = f.attrs['time_step']
                self.collision_detection = f.attrs['collision_detection']
                
                # Load bodies
                bodies_group = f['bodies']
                for body_name in bodies_group:
                    body_group = bodies_group[body_name]
                    
                    # Create body
                    body = CelestialBody(
                        name=body_group.attrs['name'],
                        mass=body_group.attrs['mass'],
                        radius=body_group.attrs['radius'],
                        position=body_group['position'][:],
                        velocity=body_group['velocity'][:],
                        color=body_group.attrs['color'],
                        density=body_group.attrs['density']
                    )
                    
                    # Set additional properties
                    body.spin_axis = body_group['spin_axis'][:]
                    body.spin_rate = body_group.attrs['spin_rate']
                    body.obliquity = body_group.attrs['obliquity']
                    
                    # Load trajectory if available
                    if 'trajectory' in body_group:
                        body.trajectory = [np.array(pos) for pos in body_group['trajectory'][:]]
                    
                    self.bodies.append(body)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        logger.info(f"Simulation loaded from {file_path}")
    
    def visualize_2d(self, ax=None, show_trajectories: bool = True, 
                    xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None) -> plt.Axes:
        """
        Visualize the current state of the simulation in 2D.
        
        Args:
            ax: Matplotlib axes to plot on (creates new figure if None)
            show_trajectories: Whether to show body trajectories
            xlim: X-axis limits as (min, max)
            ylim: Y-axis limits as (min, max)
            
        Returns:
            Matplotlib axes with the visualization
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
        # Plot each body
        for body in self.bodies:
            # Plot current position
            ax.scatter(body.position[0], body.position[1], 
                      s=np.sqrt(body.radius) * 10, color=body.color, 
                      label=body.name, alpha=0.8)
            
            # Plot trajectory if requested
            if show_trajectories and len(body.trajectory) > 1:
                trajectory = np.array(body.trajectory)
                ax.plot(trajectory[:, 0], trajectory[:, 1], 
                       color=body.color, alpha=0.3, linewidth=1)
                
            # Add text label
            ax.text(body.position[0], body.position[1], body.name, 
                   fontsize=8, ha='center', va='bottom')
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Universe Simulation - Time: {self.time/86400:.2f} days')
        
        # Set limits if provided
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
            
        # Add legend
        ax.legend(loc='upper right')
        
        return ax
    
    def visualize_3d(self, ax=None, show_trajectories: bool = True,
                    xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None,
                    zlim: Tuple[float, float] = None) -> plt.Axes:
        """
        Visualize the current state of the simulation in 3D.
        
        Args:
            ax: Matplotlib 3D axes to plot on (creates new figure if None)
            show_trajectories: Whether to show body trajectories
            xlim, ylim, zlim: Axis limits as (min, max)
            
        Returns:
            Matplotlib 3D axes with the visualization
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
        # Plot each body
        for body in self.bodies:
            # Plot current position
            ax.scatter(body.position[0], body.position[1], body.position[2],
                      s=np.sqrt(body.radius) * 10, color=body.color, 
                      label=body.name, alpha=0.8)
            
            # Plot trajectory if requested
            if show_trajectories and len(body.trajectory) > 1:
                trajectory = np.array(body.trajectory)
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                       color=body.color, alpha=0.3, linewidth=1)
                
            # Add text label
            ax.text(body.position[0], body.position[1], body.position[2], 
                   body.name, fontsize=8)
        
        # Set plot properties
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Universe Simulation - Time: {self.time/86400:.2f} days')
        
        # Set limits if provided
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if zlim:
            ax.set_zlim(zlim)
            
        # Add legend
        ax.legend(loc='upper right')
        
        return ax
    
    def create_animation(self, output_path: str = None, duration: float = 10.0, 
                        fps: int = 30, mode: str = '2d') -> None:
        """
        Create an animation of the simulation.
        
        Args:
            output_path: Path to save the animation (if None, displays instead)
            duration: Duration of the animation in seconds
            fps: Frames per second
            mode: Visualization mode ('2d' or '3d')
        """
        # Calculate number of frames
        num_frames = int(duration * fps)
        
        # Calculate time step for animation
        anim_time_step = self.time_step * (len(self.history) / num_frames)
        
        # Create figure and axes
        if mode == '2d':
            fig, ax = plt.subplots(figsize=(10, 8))
            
            def update(frame):
                ax.clear()
                frame_idx = min(int(frame * len(self.history) / num_frames), len(self.history) - 1)
                state = self.history[frame_idx]
                
                # Recreate bodies from state
                bodies = []
                for body_data in state['bodies']:
                    body = CelestialBody.from_dict(body_data)
                    bodies.append(body)
                
                # Plot bodies
                for body in bodies:
                    ax.scatter(body.position[0], body.position[1], 
                              s=np.sqrt(body.radius) * 10, color=body.color, 
                              label=body.name, alpha=0.8)
                    
                    # Add text label
                    ax.text(body.position[0], body.position[1], body.name, 
                           fontsize=8, ha='center', va='bottom')
                
                # Set plot properties
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title(f'Universe Simulation - Time: {state["time"]/86400:.2f} days')
                
                return ax
                
        elif mode == '3d':
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            def update(frame):
                ax.clear()
                frame_idx = min(int(frame * len(self.history) / num_frames), len(self.history) - 1)
                state = self.history[frame_idx]
                
                # Recreate bodies from state
                bodies = []
                for body_data in state['bodies']:
                    body = CelestialBody.from_dict(body_data)
                    bodies.append(body)
                
                # Plot bodies
                for body in bodies:
                    ax.scatter(body.position[0], body.position[1], body.position[2],
                              s=np.sqrt(body.radius) * 10, color=body.color, 
                              label=body.name, alpha=0.8)
                    
                    # Add text label
                    ax.text(body.position[0], body.position[1], body.position[2], 
                           body.name, fontsize=8)
                
                # Set plot properties
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title(f'Universe Simulation - Time: {state["time"]/86400:.2f} days')
                
                return ax
        else:
            raise ValueError(f"Unsupported visualization mode: {mode}")
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps)
        
        # Save or display animation
        if output_path:
            anim.save(output_path, writer='ffmpeg', fps=fps)
            logger.info(f"Animation saved to {output_path}")
        else:
            plt.show()
    
    def create_interactive_visualization(self) -> None:
        """Create an interactive 3D visualization using Plotly."""
        # Extract data for visualization
        data = {
            'name': [],
            'x': [],
            'y': [],
            'z': [],
            'mass': [],
            'radius': [],
            'color': []
        }
        
        for body in self.bodies:
            data['name'].append(body.name)
            data['x'].append(body.position[0])
            data['y'].append(body.position[1])
            data['z'].append(body.position[2])
            data['mass'].append(body.mass)
            data['radius'].append(body.radius)
            data['color'].append(body.color)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create 3D scatter plot
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='name',
                           size='radius', hover_name='name',
                           title=f'Universe Simulation - Time: {self.time/86400:.2f} days')
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            )
        )
        
        # Show plot
        fig.show()
    
    def create_solar_system(self) -> None:
        """Create a simplified model of our solar system."""
        # Sun
        sun = CelestialBody(
            name="Sun",
            mass=1.989e30,  # kg
            radius=6.957e8,  # m
            position=np.array([0, 0, 0]),
            velocity=np.array([0, 0, 0]),
            color='yellow'
        )
        self.add_body(sun)
        
        # Mercury
        mercury = CelestialBody(
            name="Mercury",
            mass=3.3011e23,  # kg
            radius=2.4397e6,  # m
            position=np.array([5.7909e10, 0, 0]),
            velocity=np.array([0, 4.7362e4, 0]),
            color='gray'
        )
        self.add_body(mercury)
        
        # Venus
        venus = CelestialBody(
            name="Venus",
            mass=4.8675e24,  # kg
            radius=6.0518e6,  # m
            position=np.array([1.0821e11, 0, 0]),
            velocity=np.array([0, 3.5020e4, 0]),
            color='orange'
        )
        self.add_body(venus)
        
        # Earth
        earth = CelestialBody(
            name="Earth",
            mass=5.9724e24,  # kg
            radius=6.3781e6,  # m
            position=np.array([1.4960e11, 0, 0]),
            velocity=np.array([0, 2.9783e4, 0]),
            color='blue'
        )
        self.add_body(earth)
        
        # Mars
        mars = CelestialBody(
            name="Mars",
            mass=6.4171e23,  # kg
            radius=3.3895e6,  # m
            position=np.array([2.2794e11, 0, 0]),
            velocity=np.array([0, 2.4077e4, 0]),
            color='red'
        )
        self.add_body(mars)
        
        # Jupiter
        jupiter = CelestialBody(
            name="Jupiter",
            mass=1.8982e27,  # kg
            radius=6.9911e7,  # m
            position=np.array([7.7857e11, 0, 0]),
            velocity=np.array([0, 1.3070e4, 0]),
            color='brown'
        )
        self.add_body(jupiter)
        
        # Saturn
        saturn = CelestialBody(
            name="Saturn",
            mass=5.6834e26,  # kg
            radius=5.8232e7,  # m
            position=np.array([1.4335e12, 0, 0]),
            velocity=np.array([0, 9.6724e3, 0]),
            color='gold'
        )
        self.add_body(saturn)
        
        # Uranus
        uranus = CelestialBody(
            name="Uranus",
            mass=8.6810e25,  # kg
            radius=2.5362e7,  # m
            position=np.array([2.8725e12, 0, 0]),
            velocity=np.array([0, 6.8352e3, 0]),
            color='lightblue'
        )
        self.add_body(uranus)
        
        # Neptune
        neptune = CelestialBody(
            name="Neptune",
            mass=1.0243e26,  # kg
            radius=2.4622e7,  # m
            position=np.array([4.4951e12, 0, 0]),
            velocity=np.array([0, 5.4778e3, 0]),
            color='darkblue'
        )
        self.add_body(neptune)
        
        logger.info("Solar system created")
    
    def create_binary_star_system(self, separation: float = 1.5e11) -> None:
        """Create a binary star system."""
        # Star 1
        star1 = CelestialBody(
            name="Star A",
            mass=1.5e30,  # kg (slightly less than Sun)
            radius=8e8,  # m
            position=np.array([-separation/2, 0, 0]),
            velocity=np.array([0, -1.5e4, 0]),
            color='yellow'
        )
        self.add_body(star1)
        
        # Star 2
        star2 = CelestialBody(
            name="Star B",
            mass=1.2e30,  # kg (slightly less than Sun)
            radius=7e8,  # m
            position=np.array([separation/2, 0, 0]),
            velocity=np.array([0, 1.8e4, 0]),
            color='orange'
        )
        self.add_body(star2)
        
        # Add a few planets
        planet1 = CelestialBody(
            name="Planet 1",
            mass=7e24,  # kg
            radius=7e6,  # m
            position=np.array([separation * 2, 0, 0]),
            velocity=np.array([0, 2.2e4, 0]),
            color='blue'
        )
        self.add_body(planet1)
        
        planet2 = CelestialBody(
            name="Planet 2",
            mass=9e25,  # kg
            radius=5e7,  # m
            position=np.array([separation * 3, 0, 0]),
            velocity=np.array([0, 1.8e4, 0]),
            color='green'
        )
        self.add_body(planet2)
        
        logger.info("Binary star system created")
    
    def create_galaxy_simulation(self, num_stars: int = 100, radius: float = 5e20) -> None:
        """Create a simplified galaxy simulation with many stars."""
        # Central black hole
        black_hole = CelestialBody(
            name="Central Black Hole",
            mass=1e36,  # kg (supermassive black hole)
            radius=1e12,  # m (event horizon)
            position=np.array([0, 0, 0]),
            velocity=np.array([0, 0, 0]),
            color='black'
        )
        self.add_body(black_hole)
        
        # Add stars in a spiral pattern
        for i in range(num_stars):
            # Calculate position in spiral
            angle = i * 0.1
            distance = radius * (0.1 + 0.9 * i / num_stars)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            z = (np.random.random() - 0.5) * radius * 0.1  # Thin disk
            
            # Calculate orbital velocity (simplified)
            orbital_speed = np.sqrt(gravitational_constant.value * black_hole.mass / distance)
            vx = -orbital_speed * np.sin(angle)
            vy = orbital_speed * np.cos(angle)
            vz = 0
            
            # Random star properties
            star_mass = 1e30 * (0.1 + 0.9 * np.random.random())  # 0.1 to 1 solar mass
            star_radius = 6.957e8 * (0.1 + 0.9 * np.random.random())  # 0.1 to 1 solar radius
            
            # Random color (based on mass - bigger stars are bluer)
            if star_mass > 0.8e30:
                color = 'blue'
            elif star_mass > 0.5e30:
                color = 'white'
            else:
                color = 'red'
            
            # Create star
            star = CelestialBody(
                name=f"Star {i+1}",
                mass=star_mass,
                radius=star_radius,
                position=np.array([x, y, z]),
                velocity=np.array([vx, vy, vz]),
                color=color
            )
            self.add_body(star)
        
        logger.info(f"Galaxy simulation created with {num_stars} stars")
    
    def create_expanding_universe(self, num_galaxies: int = 10, 
                                 hubble_constant: float = 70.0) -> None:
        """
        Create a simplified expanding universe simulation.
        
        Args:
            num_galaxies: Number of galaxies to create
            hubble_constant: Hubble constant in km/s/Mpc
        """
        # Convert Hubble constant to SI units (s^-1)
        H0 = hubble_constant * 1000 / (3.086e22)  # km/s/Mpc to 1/s
        
        # Create galaxies
        for i in range(num_galaxies):
            # Random position (within 100 Mpc)
            distance = 3.086e24 * np.random.random()  # Up to 100 Mpc in meters
            theta = np.random.random() * 2 * np.pi
            phi = np.random.random() * np.pi
            
            x = distance * np.sin(phi) * np.cos(theta)
            y = distance * np.sin(phi) * np.sin(theta)
            z = distance * np.cos(phi)
            
            position = np.array([x, y, z])
            
            # Velocity according to Hubble's law (v = H0 * d)
            velocity = H0 * position
            
            # Random galaxy properties
            galaxy_mass = 1e41 * (0.1 + 0.9 * np.random.random())  # Galaxy mass
            galaxy_radius = 3.086e20 * (0.1 + 0.9 * np.random.random())  # Galaxy radius
            
            # Random color
            colors = ['blue', 'white', 'yellow', 'red', 'purple']
            color = np.random.choice(colors)
            
            # Create galaxy (represented as a single body for simplicity)
            galaxy = CelestialBody(
                name=f"Galaxy {i+1}",
                mass=galaxy_mass,
                radius=galaxy_radius,
                position=position,
                velocity=velocity,
                color=color
            )
            self.add_body(galaxy)
        
        logger.info(f"Expanding universe created with {num_galaxies} galaxies")
        
        # Set cosmology
        self.cosmology = cosmo.FlatLambdaCDM(H0=hubble_constant, Om0=0.3)
    
    def calculate_cosmological_parameters(self, redshift: float) -> Dict[str, float]:
        """
        Calculate cosmological parameters at a given redshift.
        
        Args:
            redshift: Cosmological redshift
            
        Returns:
            Dictionary of cosmological parameters
        """
        # Calculate age of the universe at this redshift
        age = self.cosmology.age(redshift).value  # Gyr
        
        # Calculate lookback time
        lookback_time = self.cosmology.lookback_time(redshift).value  # Gyr
        
        # Calculate comoving distance
        comoving_distance = self.cosmology.comoving_distance(redshift).value  # Mpc
        
        # Calculate angular diameter distance
        angular_diameter_distance = self.cosmology.angular_diameter_distance(redshift).value  # Mpc
        
        # Calculate luminosity distance
        luminosity_distance = self.cosmology.luminosity_distance(redshift).value  # Mpc
        
        # Calculate Hubble parameter at this redshift
        hubble_parameter = self.cosmology.H(redshift).value  # km/s/Mpc
        
        # Calculate critical density
        critical_density = self.cosmology.critical_density(redshift).value  # g/cm^3
        
        return {
            'redshift': redshift,
            'age_gyr': age,
            'lookback_time_gyr': lookback_time,
            'comoving_distance_mpc': comoving_distance,
            'angular_diameter_distance_mpc': angular_diameter_distance,
            'luminosity_distance_mpc': luminosity_distance,
            'hubble_parameter_km_s_mpc': hubble_parameter,
            'critical_density_g_cm3': critical_density
        }
    
    def simulate_cosmic_expansion(self, time_range: List[float], 
                                 num_points: int = 100) -> pd.DataFrame:
        """
        Simulate cosmic expansion over a range of time.
        
        Args:
            time_range: Range of time in Gyr [start, end]
            num_points: Number of points to calculate
            
        Returns:
            DataFrame with cosmological parameters over time
        """
        # Convert time range to redshift range
        # Current age of the universe
        current_age = self.cosmology.age(0).value  # Gyr
        
        # Calculate redshifts corresponding to ages
        ages = np.linspace(current_age - time_range[1], current_age - time_range[0], num_points)
        redshifts = []
        
        for age in ages:
            # Find redshift corresponding to this age
            # This is an approximation using binary search
            z_min, z_max = 0, 1000
            while z_max - z_min > 0.001:
                z_mid = (z_min + z_max) / 2
                age_at_z = self.cosmology.age(z_mid).value
                if age_at_z < age:
                    z_max = z_mid
                else:
                    z_min = z_mid
            redshifts.append(z_mid)
        
        # Calculate parameters at each redshift
        data = []
        for z in redshifts:
            params = self.calculate_cosmological_parameters(z)
            data.append(params)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def plot_cosmic_expansion(self, df: pd.DataFrame) -> None:
        """
        Plot cosmic expansion parameters.
        
        Args:
            df: DataFrame with cosmological parameters
        """
        # Create subplots
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=('Scale Factor vs. Time', 
                                          'Hubble Parameter vs. Redshift',
                                          'Distance vs. Redshift',
                                          'Density vs. Redshift'))
        
        # Calculate scale factor
        df['scale_factor'] = 1 / (1 + df['redshift'])
        
        # Plot scale factor vs. time
        fig.add_trace(
            go.Scatter(x=df['age_gyr'], y=df['scale_factor'], mode='lines', name='Scale Factor'),
            row=1, col=1
        )
        
        # Plot Hubble parameter vs. redshift
        fig.add_trace(
            go.Scatter(x=df['redshift'], y=df['hubble_parameter_km_s_mpc'], mode='lines', name='Hubble Parameter'),
            row=1, col=2
        )
        
        # Plot distances vs. redshift
        fig.add_trace(
            go.Scatter(x=df['redshift'], y=df['comoving_distance_mpc'], mode='lines', name='Comoving Distance'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['redshift'], y=df['luminosity_distance_mpc'], mode='lines', name='Luminosity Distance'),
            row=2, col=1
        )
        
        # Plot critical density vs. redshift
        fig.add_trace(
            go.Scatter(x=df['redshift'], y=df['critical_density_g_cm3'], mode='lines', name='Critical Density'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Cosmic Expansion Parameters',
            height=800,
            width=1000
        )
        
        # Update axes
        fig.update_xaxes(title_text='Age (Gyr)', row=1, col=1)
        fig.update_yaxes(title_text='Scale Factor', row=1, col=1)
        
        fig.update_xaxes(title_text='Redshift', row=1, col=2)
        fig.update_yaxes(title_text='Hubble Parameter (km/s/Mpc)', row=1, col=2)
        
        fig.update_xaxes(title_text='Redshift', row=2, col=1)
        fig.update_yaxes(title_text='Distance (Mpc)', row=2, col=1)
        
        fig.update_xaxes(title_text='Redshift', row=2, col=2)
        fig.update_yaxes(title_text='Critical Density (g/cm³)', row=2, col=2)
        
        # Show plot
        fig.show()

# Example usage
if __name__ == "__main__":
    # Create simulator
    simulator = UniverseSimulator(time_step=3600)  # 1 hour time step
    
    # Create solar system
    simulator.create_solar_system()
    
    # Run simulation for 1 Earth year
    steps = int(365 * 24)  # 1 year in hours
    simulator.run_simulation(steps)
    
    # Visualize results
    plt.figure(figsize=(10, 8))
    simulator.visualize_2d()
    plt.show()