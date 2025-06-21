import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import io
import base64
from datetime import datetime
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# App metadata
APP_NAME = "oni_molecular_dynamics"
APP_DESCRIPTION = "Molecular dynamics simulation and analysis for ONI"
APP_VERSION = "1.0.0"
APP_AUTHOR = "ONI Team"
APP_CATEGORY = "Science"
APP_DEPENDENCIES = ["numpy", "matplotlib"]
APP_DEFAULT = False

class ONIMolecularDynamics:
    """
    ONI Molecular Dynamics - Simulation and analysis of molecular systems.
    
    Provides capabilities for:
    - Particle-based simulations
    - Force field calculations
    - Trajectory analysis
    - Thermodynamic property calculation
    - Visualization of molecular systems
    """
    
    def __init__(self):
        """Initialize the ONI Molecular Dynamics module."""
        self.systems = {}
        self.trajectories = {}
        self.analysis_results = {}
        self.figure_counter = 0
        
        # Default simulation parameters
        self.default_params = {
            'dt': 0.01,  # Time step
            'temperature': 300.0,  # Temperature in K
            'box_size': [10.0, 10.0, 10.0],  # Simulation box size
            'cutoff': 2.5,  # Cutoff distance for interactions
            'thermostat': 'none',  # Thermostat type
            'boundary': 'periodic'  # Boundary conditions
        }
        
        logger.info("ONI Molecular Dynamics initialized")
    
    def create_system(self, 
                     name: str,
                     n_particles: int,
                     particle_types: List[str] = None,
                     positions: np.ndarray = None,
                     velocities: np.ndarray = None,
                     masses: np.ndarray = None,
                     charges: np.ndarray = None,
                     box_size: List[float] = None) -> Dict[str, Any]:
        """
        Create a molecular system.
        
        Args:
            name: Name of the system
            n_particles: Number of particles
            particle_types: List of particle types
            positions: Initial positions (n_particles x 3)
            velocities: Initial velocities (n_particles x 3)
            masses: Particle masses
            charges: Particle charges
            box_size: Simulation box size [Lx, Ly, Lz]
            
        Returns:
            Dict[str, Any]: System information
        """
        try:
            # Set default values
            if box_size is None:
                box_size = self.default_params['box_size']
            
            if particle_types is None:
                particle_types = ['A'] * n_particles
            
            # Generate random positions if not provided
            if positions is None:
                positions = np.random.uniform(0, box_size[0], (n_particles, 3))
                for i in range(3):
                    if i < len(box_size):
                        positions[:, i] = np.random.uniform(0, box_size[i], n_particles)
            
            # Generate random velocities if not provided
            if velocities is None:
                velocities = np.random.normal(0, 1, (n_particles, 3))
                # Remove center of mass motion
                velocities -= np.mean(velocities, axis=0)
            
            # Set default masses if not provided
            if masses is None:
                masses = np.ones(n_particles)
            
            # Set default charges if not provided
            if charges is None:
                charges = np.zeros(n_particles)
            
            # Create system dictionary
            system = {
                'name': name,
                'n_particles': n_particles,
                'particle_types': particle_types,
                'positions': positions,
                'velocities': velocities,
                'forces': np.zeros_like(positions),
                'masses': masses,
                'charges': charges,
                'box_size': box_size,
                'time': 0.0,
                'step': 0,
                'potential_energy': 0.0,
                'kinetic_energy': 0.0,
                'temperature': 0.0
            }
            
            # Calculate initial energies and temperature
            self._update_energies(system)
            
            # Store the system
            self.systems[name] = system
            
            logger.info(f"Created system '{name}' with {n_particles} particles")
            return system
            
        except Exception as e:
            logger.error(f"Error creating system: {e}")
            raise
    
    def _update_energies(self, system: Dict[str, Any]) -> None:
        """
        Update energies and temperature of the system.
        
        Args:
            system: System dictionary
        """
        # Calculate kinetic energy
        velocities = system['velocities']
        masses = system['masses']
        
        # Reshape masses for broadcasting
        masses_reshaped = masses.reshape(-1, 1)
        
        # Calculate kinetic energy: 0.5 * m * v^2
        kinetic_energy = 0.5 * np.sum(masses_reshaped * np.sum(velocities**2, axis=1))
        
        # Calculate temperature: 2/3 * KE / (N * k_B)
        # Note: We're using reduced units where k_B = 1
        temperature = 2.0 * kinetic_energy / (3.0 * system['n_particles'])
        
        # Update system
        system['kinetic_energy'] = kinetic_energy
        system['temperature'] = temperature
    
    def _lennard_jones_force(self, 
                            r_vec: np.ndarray, 
                            epsilon: float = 1.0, 
                            sigma: float = 1.0, 
                            cutoff: float = 2.5) -> Tuple[np.ndarray, float]:
        """
        Calculate Lennard-Jones force and potential.
        
        Args:
            r_vec: Distance vector
            epsilon: Depth of the potential well
            sigma: Distance at which the potential is zero
            cutoff: Cutoff distance
            
        Returns:
            Tuple[np.ndarray, float]: Force vector and potential energy
        """
        r = np.linalg.norm(r_vec)
        
        # Apply cutoff
        if r > cutoff:
            return np.zeros_like(r_vec), 0.0
        
        # Calculate force and potential
        sr6 = (sigma / r)**6
        sr12 = sr6**2
        
        force = 24.0 * epsilon * (2.0 * sr12 - sr6) / r**2 * r_vec
        potential = 4.0 * epsilon * (sr12 - sr6)
        
        return force, potential
    
    def _calculate_forces(self, system: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Calculate forces between particles.
        
        Args:
            system: System dictionary
            params: Simulation parameters
        """
        positions = system['positions']
        n_particles = system['n_particles']
        box_size = system['box_size']
        cutoff = params.get('cutoff', self.default_params['cutoff'])
        boundary = params.get('boundary', self.default_params['boundary'])
        
        # Reset forces and potential energy
        forces = np.zeros_like(positions)
        potential_energy = 0.0
        
        # Calculate pairwise forces
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                # Calculate distance vector
                r_vec = positions[j] - positions[i]
                
                # Apply periodic boundary conditions if needed
                if boundary == 'periodic':
                    for k in range(3):
                        if k < len(box_size):
                            # Find the nearest image
                            r_vec[k] = r_vec[k] - box_size[k] * round(r_vec[k] / box_size[k])
                
                # Calculate force and potential
                force, potential = self._lennard_jones_force(r_vec, cutoff=cutoff)
                
                # Update forces (Newton's third law)
                forces[i] += force
                forces[j] -= force
                
                # Update potential energy
                potential_energy += potential
        
        # Update system
        system['forces'] = forces
        system['potential_energy'] = potential_energy
    
    def _velocity_verlet_step(self, system: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Perform a velocity Verlet integration step.
        
        Args:
            system: System dictionary
            params: Simulation parameters
        """
        dt = params.get('dt', self.default_params['dt'])
        positions = system['positions']
        velocities = system['velocities']
        forces = system['forces']
        masses = system['masses'].reshape(-1, 1)  # Reshape for broadcasting
        
        # Update positions: r(t+dt) = r(t) + v(t)*dt + 0.5*f(t)/m*dt^2
        positions += velocities * dt + 0.5 * forces / masses * dt**2
        
        # Store old forces
        old_forces = forces.copy()
        
        # Apply boundary conditions
        self._apply_boundary_conditions(system, params)
        
        # Calculate new forces
        self._calculate_forces(system, params)
        
        # Update velocities: v(t+dt) = v(t) + 0.5*(f(t) + f(t+dt))/m*dt
        velocities += 0.5 * (old_forces + system['forces']) / masses * dt
        
        # Apply thermostat
        self._apply_thermostat(system, params)
        
        # Update energies and temperature
        self._update_energies(system)
        
        # Update time and step
        system['time'] += dt
        system['step'] += 1
    
    def _apply_boundary_conditions(self, system: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Apply boundary conditions to the system.
        
        Args:
            system: System dictionary
            params: Simulation parameters
        """
        boundary = params.get('boundary', self.default_params['boundary'])
        positions = system['positions']
        box_size = system['box_size']
        
        if boundary == 'periodic':
            # Apply periodic boundary conditions
            for i in range(3):
                if i < len(box_size):
                    # Wrap particles back into the box
                    positions[:, i] = positions[:, i] % box_size[i]
        elif boundary == 'reflective':
            # Apply reflective boundary conditions
            for i in range(3):
                if i < len(box_size):
                    # Reflect particles at the boundaries
                    reflection = (positions[:, i] < 0) | (positions[:, i] > box_size[i])
                    positions[reflection, i] = np.where(
                        positions[reflection, i] < 0,
                        -positions[reflection, i],
                        2 * box_size[i] - positions[reflection, i]
                    )
                    # Reverse velocity component
                    system['velocities'][reflection, i] = -system['velocities'][reflection, i]
    
    def _apply_thermostat(self, system: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Apply a thermostat to control temperature.
        
        Args:
            system: System dictionary
            params: Simulation parameters
        """
        thermostat = params.get('thermostat', self.default_params['thermostat'])
        target_temp = params.get('temperature', self.default_params['temperature'])
        
        if thermostat == 'none':
            # No thermostat
            return
        elif thermostat == 'berendsen':
            # Berendsen thermostat
            tau = params.get('tau', 0.1)  # Time constant
            dt = params.get('dt', self.default_params['dt'])
            
            # Calculate current temperature
            current_temp = system['temperature']
            
            if current_temp > 0:
                # Calculate scaling factor
                lambda_factor = np.sqrt(1 + (dt / tau) * (target_temp / current_temp - 1))
                
                # Scale velocities
                system['velocities'] *= lambda_factor
        elif thermostat == 'andersen':
            # Andersen thermostat
            collision_freq = params.get('collision_freq', 0.1)
            dt = params.get('dt', self.default_params['dt'])
            
            # Probability of collision
            collision_prob = 1.0 - np.exp(-collision_freq * dt)
            
            # Randomly select particles for collision
            collisions = np.random.random(system['n_particles']) < collision_prob
            
            if np.any(collisions):
                # Generate new velocities from Maxwell-Boltzmann distribution
                std_dev = np.sqrt(target_temp / system['masses'][collisions].reshape(-1, 1))
                system['velocities'][collisions] = np.random.normal(0, std_dev)
    
    def run_simulation(self, 
                      system_name: str,
                      n_steps: int,
                      params: Dict[str, Any] = None,
                      save_trajectory: bool = True,
                      save_frequency: int = 10) -> Dict[str, Any]:
        """
        Run a molecular dynamics simulation.
        
        Args:
            system_name: Name of the system to simulate
            n_steps: Number of simulation steps
            params: Simulation parameters (optional)
            save_trajectory: Whether to save the trajectory
            save_frequency: Frequency of saving trajectory frames
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        try:
            if system_name not in self.systems:
                raise ValueError(f"System '{system_name}' not found")
            
            # Get system
            system = self.systems[system_name]
            
            # Set default parameters
            if params is None:
                params = self.default_params.copy()
            else:
                # Merge with defaults
                default_params = self.default_params.copy()
                default_params.update(params)
                params = default_params
            
            # Initialize trajectory if saving
            if save_trajectory:
                trajectory = {
                    'system_name': system_name,
                    'n_particles': system['n_particles'],
                    'box_size': system['box_size'],
                    'times': [],
                    'positions': [],
                    'velocities': [],
                    'energies': {
                        'kinetic': [],
                        'potential': [],
                        'total': []
                    },
                    'temperature': []
                }
            
            # Calculate initial forces
            self._calculate_forces(system, params)
            
            # Run simulation
            start_time = time.time()
            
            for step in range(n_steps):
                # Perform integration step
                self._velocity_verlet_step(system, params)
                
                # Save trajectory frame if needed
                if save_trajectory and step % save_frequency == 0:
                    trajectory['times'].append(system['time'])
                    trajectory['positions'].append(system['positions'].copy())
                    trajectory['velocities'].append(system['velocities'].copy())
                    trajectory['energies']['kinetic'].append(system['kinetic_energy'])
                    trajectory['energies']['potential'].append(system['potential_energy'])
                    trajectory['energies']['total'].append(
                        system['kinetic_energy'] + system['potential_energy']
                    )
                    trajectory['temperature'].append(system['temperature'])
            
            end_time = time.time()
            simulation_time = end_time - start_time
            
            # Create result dictionary
            result = {
                'system_name': system_name,
                'n_steps': n_steps,
                'params': params,
                'final_state': {
                    'time': system['time'],
                    'step': system['step'],
                    'kinetic_energy': system['kinetic_energy'],
                    'potential_energy': system['potential_energy'],
                    'total_energy': system['kinetic_energy'] + system['potential_energy'],
                    'temperature': system['temperature']
                },
                'performance': {
                    'simulation_time': simulation_time,
                    'steps_per_second': n_steps / simulation_time
                }
            }
            
            # Save trajectory
            if save_trajectory:
                trajectory_name = f"{system_name}_trajectory_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                self.trajectories[trajectory_name] = trajectory
                result['trajectory_name'] = trajectory_name
            
            logger.info(f"Simulation of '{system_name}' completed: {n_steps} steps in {simulation_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            raise
    
    def plot_trajectory(self, 
                       trajectory_name: str,
                       plot_type: str = 'energy',
                       figsize: Tuple[int, int] = (10, 6),
                       return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot trajectory data.
        
        Args:
            trajectory_name: Name of the trajectory to plot
            plot_type: Type of plot ('energy', 'temperature', 'positions', 'animation')
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            if trajectory_name not in self.trajectories:
                raise ValueError(f"Trajectory '{trajectory_name}' not found")
            
            trajectory = self.trajectories[trajectory_name]
            
            # Create figure
            if plot_type == 'positions' or plot_type == 'animation':
                # 3D plot for positions
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(figsize=figsize)
            
            if plot_type == 'energy':
                # Plot energy vs time
                times = trajectory['times']
                kinetic = trajectory['energies']['kinetic']
                potential = trajectory['energies']['potential']
                total = trajectory['energies']['total']
                
                ax.plot(times, kinetic, label='Kinetic Energy')
                ax.plot(times, potential, label='Potential Energy')
                ax.plot(times, total, label='Total Energy')
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Energy')
                ax.set_title('Energy vs Time')
                
            elif plot_type == 'temperature':
                # Plot temperature vs time
                times = trajectory['times']
                temperature = trajectory['temperature']
                
                ax.plot(times, temperature)
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Temperature')
                ax.set_title('Temperature vs Time')
                
            elif plot_type == 'positions':
                # Plot final positions
                positions = trajectory['positions'][-1]
                
                # Get particle types for coloring
                system_name = trajectory['system_name']
                if system_name in self.systems:
                    particle_types = self.systems[system_name]['particle_types']
                    
                    # Create color map
                    unique_types = list(set(particle_types))
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
                    color_map = {ptype: colors[i] for i, ptype in enumerate(unique_types)}
                    
                    # Plot particles by type
                    for ptype in unique_types:
                        mask = [i for i, pt in enumerate(particle_types) if pt == ptype]
                        ax.scatter(
                            positions[mask, 0],
                            positions[mask, 1],
                            positions[mask, 2],
                            label=ptype,
                            alpha=0.7
                        )
                else:
                    # Plot all particles with same color
                    ax.scatter(
                        positions[:, 0],
                        positions[:, 1],
                        positions[:, 2],
                        alpha=0.7
                    )
                
                # Set axis limits
                box_size = trajectory['box_size']
                ax.set_xlim(0, box_size[0])
                ax.set_ylim(0, box_size[1])
                ax.set_zlim(0, box_size[2] if len(box_size) > 2 else box_size[0])
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('Particle Positions')
                
            elif plot_type == 'animation':
                # Create animation of positions
                # For simplicity, we'll just show a few frames
                positions = trajectory['positions']
                n_frames = min(10, len(positions))
                
                # Get particle types for coloring
                system_name = trajectory['system_name']
                if system_name in self.systems:
                    particle_types = self.systems[system_name]['particle_types']
                    
                    # Create color map
                    unique_types = list(set(particle_types))
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
                    color_map = {ptype: colors[i] for i, ptype in enumerate(unique_types)}
                    
                    # Plot particles by type for the first frame
                    for ptype in unique_types:
                        mask = [i for i, pt in enumerate(particle_types) if pt == ptype]
                        ax.scatter(
                            positions[0][mask, 0],
                            positions[0][mask, 1],
                            positions[0][mask, 2],
                            label=ptype,
                            alpha=0.7
                        )
                else:
                    # Plot all particles with same color for the first frame
                    ax.scatter(
                        positions[0][:, 0],
                        positions[0][:, 1],
                        positions[0][:, 2],
                        alpha=0.7
                    )
                
                # Set axis limits
                box_size = trajectory['box_size']
                ax.set_xlim(0, box_size[0])
                ax.set_ylim(0, box_size[1])
                ax.set_zlim(0, box_size[2] if len(box_size) > 2 else box_size[0])
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'Particle Positions (Frame 1/{n_frames})')
                
                # Note: In a real implementation, we would create a proper animation
                # For now, we just show the first frame
                
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            if plot_type != 'temperature':
                ax.legend()
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting trajectory: {e}")
            raise
    
    def calculate_rdf(self, 
                     trajectory_name: str,
                     n_bins: int = 100,
                     r_max: float = None) -> Dict[str, Any]:
        """
        Calculate the radial distribution function (RDF).
        
        Args:
            trajectory_name: Name of the trajectory
            n_bins: Number of bins for the histogram
            r_max: Maximum distance to consider
            
        Returns:
            Dict[str, Any]: RDF results
        """
        try:
            if trajectory_name not in self.trajectories:
                raise ValueError(f"Trajectory '{trajectory_name}' not found")
            
            trajectory = self.trajectories[trajectory_name]
            
            # Get system information
            n_particles = trajectory['n_particles']
            box_size = trajectory['box_size']
            positions = trajectory['positions']
            
            # Set default r_max if not provided
            if r_max is None:
                r_max = min(box_size) / 2.0
            
            # Create bins
            bins = np.linspace(0, r_max, n_bins + 1)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            bin_width = bins[1] - bins[0]
            
            # Initialize histogram
            histogram = np.zeros(n_bins)
            
            # Calculate RDF for each frame
            n_frames = len(positions)
            
            for frame in range(n_frames):
                pos = positions[frame]
                
                # Calculate pairwise distances
                for i in range(n_particles):
                    for j in range(i+1, n_particles):
                        # Calculate distance vector
                        r_vec = pos[j] - pos[i]
                        
                        # Apply minimum image convention for periodic boundaries
                        for k in range(3):
                            if k < len(box_size):
                                r_vec[k] = r_vec[k] - box_size[k] * round(r_vec[k] / box_size[k])
                        
                        # Calculate distance
                        r = np.linalg.norm(r_vec)
                        
                        # Add to histogram
                        if r < r_max:
                            bin_idx = int(r / r_max * n_bins)
                            if bin_idx < n_bins:
                                histogram[bin_idx] += 2  # Count each pair twice
            
            # Normalize histogram
            # Volume of each shell: 4*pi*r^2*dr
            shell_volumes = 4.0 * np.pi * bin_centers**2 * bin_width
            
            # Average number density: N/V
            volume = np.prod(box_size)
            number_density = n_particles / volume
            
            # Expected number of particles in each shell for ideal gas
            ideal_counts = shell_volumes * number_density
            
            # Normalize by number of frames and ideal counts
            rdf = histogram / (n_frames * n_particles * ideal_counts)
            
            # Create result
            result = {
                'r': bin_centers.tolist(),
                'rdf': rdf.tolist(),
                'n_bins': n_bins,
                'r_max': r_max,
                'n_frames': n_frames
            }
            
            # Store the results
            result_name = f"{trajectory_name}_rdf"
            self.analysis_results[result_name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating RDF: {e}")
            raise
    
    def plot_rdf(self, 
                rdf_result: Dict[str, Any],
                figsize: Tuple[int, int] = (10, 6),
                return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot the radial distribution function.
        
        Args:
            rdf_result: RDF result dictionary
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            # Extract data
            r = rdf_result['r']
            rdf = rdf_result['rdf']
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot RDF
            ax.plot(r, rdf)
            
            # Add horizontal line at g(r) = 1
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('r')
            ax.set_ylabel('g(r)')
            ax.set_title('Radial Distribution Function')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting RDF: {e}")
            raise
    
    def calculate_msd(self, trajectory_name: str) -> Dict[str, Any]:
        """
        Calculate the mean square displacement (MSD).
        
        Args:
            trajectory_name: Name of the trajectory
            
        Returns:
            Dict[str, Any]: MSD results
        """
        try:
            if trajectory_name not in self.trajectories:
                raise ValueError(f"Trajectory '{trajectory_name}' not found")
            
            trajectory = self.trajectories[trajectory_name]
            
            # Get positions and times
            positions = trajectory['positions']
            times = trajectory['times']
            
            # Calculate MSD
            n_frames = len(positions)
            n_particles = trajectory['n_particles']
            
            # Initialize MSD array
            msd = np.zeros(n_frames)
            
            # Reference positions (first frame)
            ref_positions = positions[0]
            
            # Calculate MSD for each frame
            for i in range(n_frames):
                # Calculate squared displacement for each particle
                squared_disp = np.sum((positions[i] - ref_positions)**2, axis=1)
                
                # Average over particles
                msd[i] = np.mean(squared_disp)
            
            # Create result
            result = {
                'times': times,
                'msd': msd.tolist(),
                'n_frames': n_frames
            }
            
            # Store the results
            result_name = f"{trajectory_name}_msd"
            self.analysis_results[result_name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating MSD: {e}")
            raise
    
    def plot_msd(self, 
                msd_result: Dict[str, Any],
                figsize: Tuple[int, int] = (10, 6),
                return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot the mean square displacement.
        
        Args:
            msd_result: MSD result dictionary
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            # Extract data
            times = msd_result['times']
            msd = msd_result['msd']
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot MSD
            ax.plot(times, msd)
            
            # Set labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('MSD')
            ax.set_title('Mean Square Displacement')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting MSD: {e}")
            raise
    
    def create_crystal_lattice(self, 
                              name: str,
                              lattice_type: str = 'fcc',
                              lattice_constant: float = 1.0,
                              n_cells: List[int] = [4, 4, 4],
                              particle_type: str = 'A') -> Dict[str, Any]:
        """
        Create a crystal lattice system.
        
        Args:
            name: Name of the system
            lattice_type: Type of lattice ('sc', 'bcc', 'fcc', 'diamond')
            lattice_constant: Lattice constant
            n_cells: Number of unit cells in each direction
            particle_type: Type of particles
            
        Returns:
            Dict[str, Any]: System information
        """
        try:
            # Calculate box size
            box_size = [n * lattice_constant for n in n_cells]
            
            # Generate positions based on lattice type
            positions = []
            
            if lattice_type == 'sc':
                # Simple cubic
                for i in range(n_cells[0]):
                    for j in range(n_cells[1]):
                        for k in range(n_cells[2] if len(n_cells) > 2 else 1):
                            positions.append([
                                i * lattice_constant,
                                j * lattice_constant,
                                k * lattice_constant
                            ])
            
            elif lattice_type == 'bcc':
                # Body-centered cubic
                for i in range(n_cells[0]):
                    for j in range(n_cells[1]):
                        for k in range(n_cells[2] if len(n_cells) > 2 else 1):
                            # Corner atom
                            positions.append([
                                i * lattice_constant,
                                j * lattice_constant,
                                k * lattice_constant
                            ])
                            # Center atom
                            positions.append([
                                (i + 0.5) * lattice_constant,
                                (j + 0.5) * lattice_constant,
                                (k + 0.5) * lattice_constant
                            ])
            
            elif lattice_type == 'fcc':
                # Face-centered cubic
                for i in range(n_cells[0]):
                    for j in range(n_cells[1]):
                        for k in range(n_cells[2] if len(n_cells) > 2 else 1):
                            # Corner atom
                            positions.append([
                                i * lattice_constant,
                                j * lattice_constant,
                                k * lattice_constant
                            ])
                            # Face atoms
                            positions.append([
                                (i + 0.5) * lattice_constant,
                                (j + 0.5) * lattice_constant,
                                k * lattice_constant
                            ])
                            positions.append([
                                (i + 0.5) * lattice_constant,
                                j * lattice_constant,
                                (k + 0.5) * lattice_constant
                            ])
                            positions.append([
                                i * lattice_constant,
                                (j + 0.5) * lattice_constant,
                                (k + 0.5) * lattice_constant
                            ])
            
            elif lattice_type == 'diamond':
                # Diamond cubic (two interpenetrating fcc lattices)
                for i in range(n_cells[0]):
                    for j in range(n_cells[1]):
                        for k in range(n_cells[2] if len(n_cells) > 2 else 1):
                            # First fcc lattice
                            positions.append([
                                i * lattice_constant,
                                j * lattice_constant,
                                k * lattice_constant
                            ])
                            positions.append([
                                (i + 0.5) * lattice_constant,
                                (j + 0.5) * lattice_constant,
                                k * lattice_constant
                            ])
                            positions.append([
                                (i + 0.5) * lattice_constant,
                                j * lattice_constant,
                                (k + 0.5) * lattice_constant
                            ])
                            positions.append([
                                i * lattice_constant,
                                (j + 0.5) * lattice_constant,
                                (k + 0.5) * lattice_constant
                            ])
                            
                            # Second fcc lattice, shifted by (1/4, 1/4, 1/4)
                            positions.append([
                                (i + 0.25) * lattice_constant,
                                (j + 0.25) * lattice_constant,
                                (k + 0.25) * lattice_constant
                            ])
                            positions.append([
                                (i + 0.75) * lattice_constant,
                                (j + 0.75) * lattice_constant,
                                (k + 0.25) * lattice_constant
                            ])
                            positions.append([
                                (i + 0.75) * lattice_constant,
                                (j + 0.25) * lattice_constant,
                                (k + 0.75) * lattice_constant
                            ])
                            positions.append([
                                (i + 0.25) * lattice_constant,
                                (j + 0.75) * lattice_constant,
                                (k + 0.75) * lattice_constant
                            ])
            
            else:
                raise ValueError(f"Unsupported lattice type: {lattice_type}")
            
            # Convert to numpy array
            positions = np.array(positions)
            
            # Count particles
            n_particles = len(positions)
            
            # Create particle types
            particle_types = [particle_type] * n_particles
            
            # Create system
            return self.create_system(
                name=name,
                n_particles=n_particles,
                particle_types=particle_types,
                positions=positions,
                box_size=box_size
            )
            
        except Exception as e:
            logger.error(f"Error creating crystal lattice: {e}")
            raise
    
    def create_liquid_system(self, 
                            name: str,
                            n_particles: int,
                            density: float = 0.8,
                            temperature: float = 1.0,
                            particle_type: str = 'A') -> Dict[str, Any]:
        """
        Create a liquid system with random positions.
        
        Args:
            name: Name of the system
            n_particles: Number of particles
            density: Number density
            temperature: Initial temperature
            particle_type: Type of particles
            
        Returns:
            Dict[str, Any]: System information
        """
        try:
            # Calculate box size
            volume = n_particles / density
            box_length = volume ** (1/3)
            box_size = [box_length, box_length, box_length]
            
            # Create system with random positions
            system = self.create_system(
                name=name,
                n_particles=n_particles,
                particle_types=[particle_type] * n_particles,
                box_size=box_size
            )
            
            # Set initial velocities to match temperature
            velocities = np.random.normal(0, np.sqrt(temperature), (n_particles, 3))
            
            # Remove center of mass motion
            velocities -= np.mean(velocities, axis=0)
            
            # Update system
            system['velocities'] = velocities
            
            # Update energies and temperature
            self._update_energies(system)
            
            return system
            
        except Exception as e:
            logger.error(f"Error creating liquid system: {e}")
            raise
    
    def equilibrate_system(self, 
                          system_name: str,
                          n_steps: int,
                          target_temperature: float = None,
                          params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Equilibrate a system to a target temperature.
        
        Args:
            system_name: Name of the system to equilibrate
            n_steps: Number of equilibration steps
            target_temperature: Target temperature
            params: Simulation parameters (optional)
            
        Returns:
            Dict[str, Any]: Equilibration results
        """
        try:
            if system_name not in self.systems:
                raise ValueError(f"System '{system_name}' not found")
            
            # Set default parameters
            if params is None:
                params = self.default_params.copy()
            else:
                # Merge with defaults
                default_params = self.default_params.copy()
                default_params.update(params)
                params = default_params
            
            # Set target temperature
            if target_temperature is not None:
                params['temperature'] = target_temperature
            
            # Set thermostat
            params['thermostat'] = 'berendsen'
            
            # Run equilibration
            result = self.run_simulation(
                system_name=system_name,
                n_steps=n_steps,
                params=params,
                save_trajectory=True
            )
            
            # Add equilibration-specific information
            result['equilibration'] = {
                'target_temperature': params['temperature'],
                'initial_temperature': self.systems[system_name]['temperature'],
                'final_temperature': self.systems[system_name]['temperature']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error equilibrating system: {e}")
            raise
    
    def run(self, command: str = None, **kwargs) -> Any:
        """
        Run a command or return help information.
        
        Args:
            command: Command to run (optional)
            **kwargs: Additional arguments
            
        Returns:
            Any: Command result or help information
        """
        if command is None:
            # Return help information
            return self.help()
        
        # Parse and execute command
        try:
            # Check if command is a method name
            if hasattr(self, command) and callable(getattr(self, command)):
                method = getattr(self, command)
                return method(**kwargs)
            
            # Otherwise, try to evaluate as a Python expression
            # This is potentially dangerous and should be used with caution
            return eval(command)
            
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return f"Error: {str(e)}"
    
    def help(self) -> str:
        """Return help information about the ONI Molecular Dynamics module."""
        help_text = """
        ONI Molecular Dynamics - Simulation and analysis of molecular systems
        
        Available methods:
        - create_system(name, n_particles, ...): Create a molecular system
        - create_crystal_lattice(name, lattice_type, ...): Create a crystal lattice system
        - create_liquid_system(name, n_particles, ...): Create a liquid system
        - run_simulation(system_name, n_steps, ...): Run a molecular dynamics simulation
        - equilibrate_system(system_name, n_steps, ...): Equilibrate a system to a target temperature
        - plot_trajectory(trajectory_name, plot_type, ...): Plot trajectory data
        - calculate_rdf(trajectory_name, ...): Calculate the radial distribution function
        - plot_rdf(rdf_result, ...): Plot the radial distribution function
        - calculate_msd(trajectory_name): Calculate the mean square displacement
        - plot_msd(msd_result, ...): Plot the mean square displacement
        
        For more information on a specific method, use help(ONIMolecularDynamics.method_name)
        """
        return help_text
    
    def cleanup(self):
        """Clean up resources."""
        # Close all matplotlib figures
        plt.close('all')
        
        # Clear systems and trajectories
        self.systems.clear()
        self.trajectories.clear()
        self.analysis_results.clear()
        
        logger.info("ONI Molecular Dynamics cleaned up")

# Example usage
if __name__ == "__main__":
    md = ONIMolecularDynamics()
    
    # Create a simple system
    system = md.create_liquid_system(
        name="liquid_argon",
        n_particles=100,
        density=0.8,
        temperature=1.0
    )
    
    # Run a short simulation
    result = md.run_simulation(
        system_name="liquid_argon",
        n_steps=1000,
        save_trajectory=True
    )
    
    # Plot energy
    trajectory_name = result['trajectory_name']
    energy_plot = md.plot_trajectory(trajectory_name, 'energy')
    
    # Calculate RDF
    rdf = md.calculate_rdf(trajectory_name)
    
    # Plot RDF
    rdf_plot = md.plot_rdf(rdf)
    
    print(f"Simulation completed: {result['n_steps']} steps")
    print(f"Final temperature: {result['final_state']['temperature']:.2f}")
    print(f"Performance: {result['performance']['steps_per_second']:.2f} steps/second")