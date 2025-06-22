import unittest
import numpy as np
import os
import tempfile
from tools.universe_simulator import CelestialBody, UniverseSimulator

class TestUniverseSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = UniverseSimulator(time_step=3600.0)  # 1 hour time step
        
    def test_celestial_body_creation(self):
        """Test creation of a celestial body with proper properties."""
        body = CelestialBody(
            name="Earth",
            mass=5.97e24,
            radius=6.37e6,
            position=np.array([1.5e11, 0, 0]),
            velocity=np.array([0, 2.98e4, 0]),
            color="blue"
        )
        
        self.assertEqual(body.name, "Earth")
        self.assertEqual(body.mass, 5.97e24)
        self.assertEqual(body.radius, 6.37e6)
        np.testing.assert_array_equal(body.position, np.array([1.5e11, 0, 0]))
        np.testing.assert_array_equal(body.velocity, np.array([0, 2.98e4, 0]))
        self.assertEqual(body.color, "blue")
        
    def test_add_body(self):
        """Test adding a body to the simulator."""
        body = CelestialBody(
            name="Earth",
            mass=5.97e24,
            radius=6.37e6,
            position=np.array([1.5e11, 0, 0]),
            velocity=np.array([0, 2.98e4, 0]),
            color="blue"
        )
        
        self.simulator.add_body(body)
        self.assertEqual(len(self.simulator.bodies), 1)
        self.assertEqual(self.simulator.bodies[0].name, "Earth")
        
    def test_remove_body(self):
        """Test removing a body from the simulator."""
        body = CelestialBody(
            name="Earth",
            mass=5.97e24,
            radius=6.37e6,
            position=np.array([1.5e11, 0, 0]),
            velocity=np.array([0, 2.98e4, 0]),
            color="blue"
        )
        
        self.simulator.add_body(body)
        self.assertEqual(len(self.simulator.bodies), 1)
        
        result = self.simulator.remove_body("Earth")
        self.assertTrue(result)
        self.assertEqual(len(self.simulator.bodies), 0)
        
        # Test removing non-existent body
        result = self.simulator.remove_body("Mars")
        self.assertFalse(result)
        
    def test_update(self):
        """Test updating the simulation by one time step."""
        # Create a simple two-body system (Sun and Earth)
        sun = CelestialBody(
            name="Sun",
            mass=1.989e30,
            radius=6.96e8,
            position=np.array([0, 0, 0]),
            velocity=np.array([0, 0, 0]),
            color="yellow"
        )
        
        earth = CelestialBody(
            name="Earth",
            mass=5.97e24,
            radius=6.37e6,
            position=np.array([1.5e11, 0, 0]),
            velocity=np.array([0, 2.98e4, 0]),
            color="blue"
        )
        
        self.simulator.add_body(sun)
        self.simulator.add_body(earth)
        
        # Store initial position
        initial_position = earth.position.copy()
        
        # Update simulation
        self.simulator.update()
        
        # Check that Earth has moved
        self.assertFalse(np.array_equal(earth.position, initial_position))
        
        # Check that time has advanced
        self.assertEqual(self.simulator.time, 3600.0)
        
    def test_gravitational_force(self):
        """Test calculation of gravitational force between bodies."""
        body1 = CelestialBody(
            name="Body1",
            mass=1.0e20,
            radius=1.0e6,
            position=np.array([0, 0, 0]),
            velocity=np.array([0, 0, 0]),
            color="red"
        )
        
        body2 = CelestialBody(
            name="Body2",
            mass=1.0e20,
            radius=1.0e6,
            position=np.array([1.0e7, 0, 0]),  # 10 million meters away
            velocity=np.array([0, 0, 0]),
            color="blue"
        )
        
        force = body1.calculate_gravitational_force(body2)
        
        # Force should be in the positive x direction
        self.assertTrue(force[0] > 0)
        self.assertEqual(force[1], 0)
        self.assertEqual(force[2], 0)
        
    def test_collision_detection(self):
        """Test collision detection and handling."""
        # Create two bodies that will collide
        body1 = CelestialBody(
            name="Body1",
            mass=1.0e20,
            radius=5.0e6,
            position=np.array([0, 0, 0]),
            velocity=np.array([1000, 0, 0]),  # Moving toward body2
            color="red"
        )
        
        body2 = CelestialBody(
            name="Body2",
            mass=1.0e20,
            radius=5.0e6,
            position=np.array([1.1e7, 0, 0]),  # Just outside collision range
            velocity=np.array([-1000, 0, 0]),  # Moving toward body1
            color="blue"
        )
        
        simulator = UniverseSimulator(time_step=1.0, collision_detection=True)
        simulator.add_body(body1)
        simulator.add_body(body2)
        
        # Run simulation until collision
        for _ in range(10):
            simulator.update()
            
            # If bodies have merged, break
            if len(simulator.bodies) < 2:
                break
        
        # Check that bodies have merged
        self.assertEqual(len(simulator.bodies), 1)
        self.assertEqual(simulator.bodies[0].mass, 2.0e20)  # Sum of masses
        
    def test_save_and_load_simulation(self):
        """Test saving and loading a simulation."""
        # Create a simple system
        sun = CelestialBody(
            name="Sun",
            mass=1.989e30,
            radius=6.96e8,
            position=np.array([0, 0, 0]),
            velocity=np.array([0, 0, 0]),
            color="yellow"
        )
        
        earth = CelestialBody(
            name="Earth",
            mass=5.97e24,
            radius=6.37e6,
            position=np.array([1.5e11, 0, 0]),
            velocity=np.array([0, 2.98e4, 0]),
            color="blue"
        )
        
        self.simulator.add_body(sun)
        self.simulator.add_body(earth)
        
        # Save simulation to temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        self.simulator.save_simulation(temp_path)
        
        # Create a new simulator and load the saved simulation
        new_simulator = UniverseSimulator()
        new_simulator.load_simulation(temp_path)
        
        # Check that bodies were loaded correctly
        self.assertEqual(len(new_simulator.bodies), 2)
        self.assertEqual(new_simulator.bodies[0].name, "Sun")
        self.assertEqual(new_simulator.bodies[1].name, "Earth")
        
        # Clean up
        os.unlink(temp_path)
        
    def test_create_solar_system(self):
        """Test creation of a solar system."""
        self.simulator.create_solar_system()
        
        # Check that all planets were created
        self.assertEqual(len(self.simulator.bodies), 9)  # Sun + 8 planets
        
        # Check that the Sun is at the center
        sun = next(body for body in self.simulator.bodies if body.name == "Sun")
        np.testing.assert_array_equal(sun.position, np.array([0, 0, 0]))
        
    def test_create_binary_star_system(self):
        """Test creation of a binary star system."""
        self.simulator.create_binary_star_system()
        
        # Check that the system was created
        self.assertEqual(len(self.simulator.bodies), 4)  # 2 stars + 2 planets
        
        # Check that we have two stars
        stars = [body for body in self.simulator.bodies if "Star" in body.name]
        self.assertEqual(len(stars), 2)
        
    def test_create_galaxy_simulation(self):
        """Test creation of a galaxy simulation."""
        self.simulator.create_galaxy_simulation(num_stars=10)
        
        # Check that the system was created
        self.assertEqual(len(self.simulator.bodies), 11)  # Central black hole + 10 stars
        
        # Check that we have a black hole
        black_hole = next(body for body in self.simulator.bodies if "Black Hole" in body.name)
        self.assertTrue(black_hole.mass > 1e30)  # Black hole should be massive

if __name__ == '__main__':
    unittest.main()