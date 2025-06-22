import unittest
import os
import tempfile
import shutil
from tools.navigation import NavigationTool

class TestNavigationTool(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.navigation_tool = NavigationTool(cache_dir=self.test_dir)
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_initialization(self):
        """Test that NavigationTool initializes correctly."""
        self.assertEqual(self.navigation_tool.cache_dir, self.test_dir)
        self.assertIsNotNone(self.navigation_tool.geocoder)
        
    def test_geocode(self):
        """Test geocoding a location."""
        result = self.navigation_tool.geocode("New York, NY")
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["location"], "New York, NY")
        self.assertIn("latitude", result)
        self.assertIn("longitude", result)
        self.assertIn("address", result)
        
        # Check that coordinates are reasonable for New York
        self.assertAlmostEqual(result["latitude"], 40.7, delta=1.0)
        self.assertAlmostEqual(result["longitude"], -74.0, delta=1.0)
        
    def test_reverse_geocode(self):
        """Test reverse geocoding coordinates."""
        # Coordinates for New York City
        latitude = 40.7128
        longitude = -74.0060
        
        result = self.navigation_tool.reverse_geocode(latitude, longitude)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["latitude"], latitude)
        self.assertEqual(result["longitude"], longitude)
        self.assertIn("address", result)
        
        # Check that address contains "New York"
        self.assertIn("New York", result["address"])
        
    def test_get_route(self):
        """Test getting a route between two locations."""
        origin = "New York, NY"
        destination = "Philadelphia, PA"
        
        result = self.navigation_tool.get_route(origin, destination, "driving")
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["origin"], origin)
        self.assertEqual(result["destination"], destination)
        self.assertEqual(result["mode"], "driving")
        self.assertIn("routes", result)
        self.assertGreater(len(result["routes"]), 0)
        
        # Check route details
        route = result["routes"][0]
        self.assertIn("coordinates", route)
        self.assertIn("length_meters", route)
        self.assertIn("duration_minutes", route)
        
        # Check that route length is reasonable (80-120 miles)
        self.assertGreater(route["length_meters"], 80 * 1609)  # 80 miles in meters
        self.assertLess(route["length_meters"], 120 * 1609)  # 120 miles in meters
        
    def test_create_map(self):
        """Test creating a map."""
        # New York City coordinates
        center = (40.7128, -74.0060)
        
        result = self.navigation_tool.create_map(center, zoom=12)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["center"], center)
        self.assertEqual(result["zoom"], 12)
        self.assertIn("map_file", result)
        self.assertIn("map_object", result)
        
        # Check that map file was created
        self.assertTrue(os.path.exists(result["map_file"]))
        
    def test_add_marker(self):
        """Test adding a marker to a map."""
        # Create a map first
        map_result = self.navigation_tool.create_map((40.7128, -74.0060), zoom=12)
        self.assertTrue(map_result["success"])
        
        # Add a marker
        location = (40.7128, -74.0060)
        popup = "New York City"
        icon = "info"
        
        result = self.navigation_tool.add_marker(
            map_result["map_object"],
            location,
            popup,
            icon
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["location"], location)
        self.assertEqual(result["popup"], popup)
        self.assertEqual(result["icon"], icon)
        
    def test_add_route_to_map(self):
        """Test adding a route to a map."""
        # Get a route first
        route_result = self.navigation_tool.get_route("New York, NY", "Philadelphia, PA", "driving")
        self.assertTrue(route_result["success"])
        
        # Create a map
        map_result = self.navigation_tool.create_map()
        self.assertTrue(map_result["success"])
        
        # Add route to map
        result = self.navigation_tool.add_route_to_map(
            map_result["map_object"],
            route_result
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["route_count"], len(route_result["routes"]))
        
    def test_get_nearby_places(self):
        """Test getting nearby places."""
        # New York City coordinates
        latitude = 40.7128
        longitude = -74.0060
        
        result = self.navigation_tool.get_nearby_places(
            latitude,
            longitude,
            category="amenity",
            radius=1000
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["category"], "amenity")
        self.assertEqual(result["radius"], 1000)
        self.assertIn("places", result)
        self.assertIn("count", result)
        
        # Should find at least some places in NYC
        self.assertGreater(result["count"], 0)
        
    def test_get_directions(self):
        """Test getting turn-by-turn directions."""
        # Get a route first
        route_result = self.navigation_tool.get_route("New York, NY", "Philadelphia, PA", "driving")
        self.assertTrue(route_result["success"])
        
        # Get directions
        result = self.navigation_tool.get_directions(route_result)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertIn("directions", result)
        self.assertIn("total_distance", result)
        self.assertIn("total_duration", result)
        
        # Should have multiple direction steps
        self.assertGreater(len(result["directions"]), 5)
        
        # Check direction step format
        step = result["directions"][0]
        self.assertIn("instruction", step)
        self.assertIn("distance_meters", step)
        self.assertIn("duration_seconds", step)
        self.assertIn("latitude", step)
        self.assertIn("longitude", step)
        
    def test_create_navigation_map(self):
        """Test creating a complete navigation map."""
        origin = "New York, NY"
        destination = "Philadelphia, PA"
        mode = "driving"
        
        result = self.navigation_tool.create_navigation_map(origin, destination, mode)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["origin"], origin)
        self.assertEqual(result["destination"], destination)
        self.assertEqual(result["mode"], mode)
        self.assertIn("map_file", result)
        self.assertIn("route", result)
        
        # Check that map file was created
        self.assertTrue(os.path.exists(result["map_file"]))

if __name__ == '__main__':
    unittest.main()