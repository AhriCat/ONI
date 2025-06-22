import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import tempfile
import requests
import folium
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopy
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import io
import base64
from datetime import datetime
import polyline
import webbrowser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NavigationTool:
    """Navigation tool for ONI using OpenStreetMap."""
    
    def __init__(self, cache_dir: Optional[str] = None, user_agent: str = "ONI_Navigation_Tool/1.0"):
        """
        Initialize navigation tool.
        
        Args:
            cache_dir: Directory for caching map data (default: temporary directory)
            user_agent: User agent for API requests
        """
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "oni_navigation_cache")
        self.user_agent = user_agent
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize geocoder
        self.geocoder = Nominatim(user_agent=self.user_agent)
        
        # Initialize route cache
        self.route_cache = {}
        
        # Set up OSMnx
        ox.config(use_cache=True, log_console=False, cache_folder=self.cache_dir)
        
        logger.info(f"Navigation tool initialized with cache directory: {self.cache_dir}")
    
    def geocode(self, location: str) -> Dict[str, Any]:
        """
        Convert a location string to coordinates.
        
        Args:
            location: Location string (e.g., "New York, NY")
            
        Returns:
            Dictionary with geocoding result
        """
        try:
            # Check cache first
            cache_file = os.path.join(self.cache_dir, f"geocode_{location.replace(' ', '_')}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Geocode location
            result = self.geocoder.geocode(location, exactly_one=True)
            
            if not result:
                return {
                    "success": False,
                    "error": f"Location not found: {location}"
                }
            
            # Format result
            geocode_result = {
                "success": True,
                "location": location,
                "latitude": result.latitude,
                "longitude": result.longitude,
                "address": result.address
            }
            
            # Cache result
            with open(cache_file, 'w') as f:
                json.dump(geocode_result, f)
            
            logger.info(f"Geocoded location: {location}")
            return geocode_result
            
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def reverse_geocode(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Convert coordinates to an address.
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Dictionary with reverse geocoding result
        """
        try:
            # Check cache first
            cache_file = os.path.join(self.cache_dir, f"reverse_geocode_{latitude}_{longitude}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Reverse geocode coordinates
            result = self.geocoder.reverse((latitude, longitude), exactly_one=True)
            
            if not result:
                return {
                    "success": False,
                    "error": f"Address not found for coordinates: {latitude}, {longitude}"
                }
            
            # Format result
            reverse_geocode_result = {
                "success": True,
                "latitude": latitude,
                "longitude": longitude,
                "address": result.address
            }
            
            # Cache result
            with open(cache_file, 'w') as f:
                json.dump(reverse_geocode_result, f)
            
            logger.info(f"Reverse geocoded coordinates: {latitude}, {longitude}")
            return reverse_geocode_result
            
        except Exception as e:
            logger.error(f"Reverse geocoding error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_route(self, origin: str, destination: str, 
                 mode: str = "driving", alternatives: bool = False) -> Dict[str, Any]:
        """
        Get a route between two locations.
        
        Args:
            origin: Origin location string
            destination: Destination location string
            mode: Transportation mode ('driving', 'walking', 'cycling', 'transit')
            alternatives: Whether to return alternative routes
            
        Returns:
            Dictionary with route information
        """
        try:
            # Check cache first
            cache_key = f"{origin}_{destination}_{mode}_{alternatives}"
            if cache_key in self.route_cache:
                return self.route_cache[cache_key]
            
            # Geocode origin and destination
            origin_result = self.geocode(origin)
            if not origin_result.get("success", False):
                return origin_result
            
            destination_result = self.geocode(destination)
            if not destination_result.get("success", False):
                return destination_result
            
            # Get coordinates
            origin_coords = (origin_result["latitude"], origin_result["longitude"])
            dest_coords = (destination_result["latitude"], destination_result["longitude"])
            
            # Get network type based on mode
            if mode == "driving":
                network_type = "drive"
            elif mode == "walking":
                network_type = "walk"
            elif mode == "cycling":
                network_type = "bike"
            else:
                network_type = "drive"  # Default
            
            # Get graph for the area
            try:
                # Get a graph that covers both points
                center_lat = (origin_coords[0] + dest_coords[0]) / 2
                center_lng = (origin_coords[1] + dest_coords[1]) / 2
                
                # Calculate distance between points
                distance = geodesic(origin_coords, dest_coords).kilometers
                
                # Add buffer to ensure graph covers the route
                buffer = max(distance * 0.5, 1.0)  # At least 1 km buffer
                
                # Get graph
                G = ox.graph_from_point((center_lat, center_lng), dist=buffer * 1000, network_type=network_type)
                
                # Get nearest nodes to origin and destination
                origin_node = ox.distance.nearest_nodes(G, origin_coords[1], origin_coords[0])
                dest_node = ox.distance.nearest_nodes(G, dest_coords[1], dest_coords[0])
                
                # Calculate route
                if alternatives:
                    # Get k-shortest paths
                    routes = list(nx.shortest_simple_paths(G, origin_node, dest_node, weight='length'))
                    # Limit to 3 alternatives
                    routes = routes[:3]
                else:
                    # Get shortest path
                    route = nx.shortest_path(G, origin_node, dest_node, weight='length')
                    routes = [route]
                
                # Process routes
                processed_routes = []
                for i, route in enumerate(routes):
                    # Get route coordinates
                    route_coords = []
                    for node in route:
                        y = G.nodes[node]['y']  # latitude
                        x = G.nodes[node]['x']  # longitude
                        route_coords.append((y, x))
                    
                    # Calculate route length
                    length = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length'))
                    
                    # Calculate estimated duration
                    if mode == "driving":
                        speed = 50  # km/h
                    elif mode == "walking":
                        speed = 5  # km/h
                    elif mode == "cycling":
                        speed = 15  # km/h
                    else:
                        speed = 50  # km/h
                    
                    duration = (length / 1000) / speed * 60  # minutes
                    
                    processed_routes.append({
                        "index": i,
                        "coordinates": route_coords,
                        "length_meters": length,
                        "duration_minutes": duration,
                        "node_count": len(route)
                    })
                
                # Create result
                route_result = {
                    "success": True,
                    "origin": origin,
                    "destination": destination,
                    "origin_coords": origin_coords,
                    "destination_coords": dest_coords,
                    "mode": mode,
                    "routes": processed_routes
                }
                
                # Cache result
                self.route_cache[cache_key] = route_result
                
                logger.info(f"Calculated route from {origin} to {destination} ({mode})")
                return route_result
                
            except Exception as e:
                logger.error(f"Route calculation error: {e}")
                
                # Fallback to OSRM API
                return self._get_route_osrm(origin_coords, dest_coords, mode, alternatives)
                
        except Exception as e:
            logger.error(f"Route error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_route_osrm(self, origin_coords: Tuple[float, float], 
                       dest_coords: Tuple[float, float], 
                       mode: str = "driving", 
                       alternatives: bool = False) -> Dict[str, Any]:
        """
        Get a route using the OSRM API (fallback method).
        
        Args:
            origin_coords: Origin coordinates (latitude, longitude)
            dest_coords: Destination coordinates (latitude, longitude)
            mode: Transportation mode
            alternatives: Whether to return alternative routes
            
        Returns:
            Dictionary with route information
        """
        try:
            # Map mode to OSRM profile
            if mode == "driving":
                profile = "car"
            elif mode == "walking":
                profile = "foot"
            elif mode == "cycling":
                profile = "bike"
            else:
                profile = "car"  # Default
            
            # Build OSRM API URL
            base_url = "https://router.project-osrm.org/route/v1"
            coords = f"{origin_coords[1]},{origin_coords[0]};{dest_coords[1]},{dest_coords[0]}"
            options = f"alternatives={'true' if alternatives else 'false'}&overview=full&steps=true"
            url = f"{base_url}/{profile}/{coords}?{options}"
            
            # Make request
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data["code"] != "Ok":
                return {
                    "success": False,
                    "error": f"OSRM API error: {data['code']}"
                }
            
            # Process routes
            processed_routes = []
            for i, route in enumerate(data["routes"]):
                # Decode polyline
                geometry = polyline.decode(route["geometry"])
                route_coords = [(lat, lng) for lat, lng in geometry]
                
                # Get route details
                distance = route["distance"]  # meters
                duration = route["duration"] / 60  # minutes
                
                processed_routes.append({
                    "index": i,
                    "coordinates": route_coords,
                    "length_meters": distance,
                    "duration_minutes": duration,
                    "node_count": len(route_coords)
                })
            
            # Create result
            route_result = {
                "success": True,
                "origin_coords": origin_coords,
                "destination_coords": dest_coords,
                "mode": mode,
                "routes": processed_routes,
                "source": "osrm"
            }
            
            logger.info(f"Got route from OSRM API")
            return route_result
            
        except Exception as e:
            logger.error(f"OSRM API error: {e}")
            return {
                "success": False,
                "error": f"OSRM API error: {str(e)}"
            }
    
    def create_map(self, center: Optional[Tuple[float, float]] = None, 
                  zoom: int = 10) -> Dict[str, Any]:
        """
        Create an interactive map.
        
        Args:
            center: Center coordinates (latitude, longitude)
            zoom: Initial zoom level
            
        Returns:
            Dictionary with map information
        """
        try:
            # Use default center if not provided
            if center is None:
                center = (40.7128, -74.0060)  # New York City
            
            # Create map
            m = folium.Map(location=center, zoom_start=zoom, control_scale=True)
            
            # Add tile layers
            folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(m)
            folium.TileLayer('Stamen Terrain', name='Terrain').add_to(m)
            folium.TileLayer('Stamen Toner', name='Toner').add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save map to HTML file
            map_file = os.path.join(self.cache_dir, f"map_{int(time.time())}.html")
            m.save(map_file)
            
            logger.info(f"Created map centered at {center}")
            return {
                "success": True,
                "center": center,
                "zoom": zoom,
                "map_file": map_file,
                "map_object": m
            }
            
        except Exception as e:
            logger.error(f"Map creation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_marker(self, map_object, location: Tuple[float, float], 
                  popup: Optional[str] = None, 
                  icon: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a marker to a map.
        
        Args:
            map_object: Folium map object
            location: Marker location (latitude, longitude)
            popup: Popup text (optional)
            icon: Icon name (optional)
            
        Returns:
            Dictionary with marker information
        """
        try:
            # Create marker
            if icon:
                marker = folium.Marker(
                    location=location,
                    popup=popup,
                    icon=folium.Icon(icon=icon)
                )
            else:
                marker = folium.Marker(
                    location=location,
                    popup=popup
                )
            
            # Add marker to map
            marker.add_to(map_object)
            
            logger.info(f"Added marker at {location}")
            return {
                "success": True,
                "location": location,
                "popup": popup,
                "icon": icon
            }
            
        except Exception as e:
            logger.error(f"Marker error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_route_to_map(self, map_object, route: Dict[str, Any], 
                        color: str = 'blue', weight: int = 5) -> Dict[str, Any]:
        """
        Add a route to a map.
        
        Args:
            map_object: Folium map object
            route: Route dictionary from get_route
            color: Line color
            weight: Line weight
            
        Returns:
            Dictionary with route information
        """
        try:
            if not route.get("success", False):
                return {
                    "success": False,
                    "error": "Invalid route"
                }
            
            # Add origin and destination markers
            origin_coords = route["origin_coords"]
            dest_coords = route["destination_coords"]
            
            self.add_marker(
                map_object=map_object,
                location=origin_coords,
                popup=f"Origin: {route.get('origin', 'Start')}",
                icon="play"
            )
            
            self.add_marker(
                map_object=map_object,
                location=dest_coords,
                popup=f"Destination: {route.get('destination', 'End')}",
                icon="stop"
            )
            
            # Add route lines
            for i, route_data in enumerate(route["routes"]):
                # Get route coordinates
                route_coords = route_data["coordinates"]
                
                # Calculate route details
                distance_km = route_data["length_meters"] / 1000
                duration_min = route_data["duration_minutes"]
                
                # Create line
                route_line = folium.PolyLine(
                    locations=route_coords,
                    color=color if i == 0 else f"{'red' if i == 1 else 'green' if i == 2 else 'purple'}",
                    weight=weight if i == 0 else weight - 2,
                    opacity=0.8 if i == 0 else 0.6,
                    popup=f"Route {i+1}: {distance_km:.1f} km, {duration_min:.1f} min"
                )
                
                # Add line to map
                route_line.add_to(map_object)
            
            # Fit map to route bounds
            all_coords = []
            for route_data in route["routes"]:
                all_coords.extend(route_data["coordinates"])
            
            if all_coords:
                map_object.fit_bounds(all_coords)
            
            logger.info(f"Added route to map")
            return {
                "success": True,
                "route_count": len(route["routes"])
            }
            
        except Exception as e:
            logger.error(f"Route mapping error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_map(self, map_object, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Save a map to an HTML file.
        
        Args:
            map_object: Folium map object
            file_path: Path to save the map (optional)
            
        Returns:
            Dictionary with save result
        """
        try:
            # Generate file path if not provided
            if file_path is None:
                file_path = os.path.join(self.cache_dir, f"map_{int(time.time())}.html")
            
            # Save map
            map_object.save(file_path)
            
            logger.info(f"Saved map to {file_path}")
            return {
                "success": True,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Map save error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def open_map(self, file_path: str) -> Dict[str, Any]:
        """
        Open a map in the default web browser.
        
        Args:
            file_path: Path to the map HTML file
            
        Returns:
            Dictionary with open result
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"Map file not found: {file_path}"
                }
            
            # Open in browser
            webbrowser.open(f"file://{os.path.abspath(file_path)}")
            
            logger.info(f"Opened map in browser: {file_path}")
            return {
                "success": True,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Map open error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_nearby_places(self, latitude: float, longitude: float, 
                         category: str = "amenity", radius: int = 1000) -> Dict[str, Any]:
        """
        Get nearby places of interest.
        
        Args:
            latitude: Latitude
            longitude: Longitude
            category: OSM category (e.g., 'amenity', 'shop', 'tourism')
            radius: Search radius in meters
            
        Returns:
            Dictionary with nearby places
        """
        try:
            # Get places using OSMnx
            tags = {category: True}
            gdf = ox.geometries_from_point((latitude, longitude), tags=tags, dist=radius)
            
            if gdf.empty:
                return {
                    "success": True,
                    "places": [],
                    "count": 0
                }
            
            # Process results
            places = []
            for idx, row in gdf.iterrows():
                # Get place name
                name = row.get('name', 'Unnamed')
                
                # Get place type
                place_type = None
                for tag in tags:
                    if tag in row:
                        place_type = row[tag]
                        break
                
                # Get coordinates
                if 'geometry' in row:
                    geom = row['geometry']
                    if hasattr(geom, 'centroid'):
                        point = geom.centroid
                        lat, lng = point.y, point.x
                    else:
                        lat, lng = geom.y, geom.x
                else:
                    continue
                
                places.append({
                    "name": name,
                    "type": place_type,
                    "category": category,
                    "latitude": lat,
                    "longitude": lng,
                    "tags": {k: v for k, v in row.items() if k not in ['geometry']}
                })
            
            logger.info(f"Found {len(places)} nearby {category} places")
            return {
                "success": True,
                "places": places,
                "count": len(places),
                "category": category,
                "radius": radius
            }
            
        except Exception as e:
            logger.error(f"Nearby places error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_nearby_places_to_map(self, map_object, latitude: float, longitude: float,
                               category: str = "amenity", radius: int = 1000) -> Dict[str, Any]:
        """
        Add nearby places to a map.
        
        Args:
            map_object: Folium map object
            latitude: Latitude
            longitude: Longitude
            category: OSM category
            radius: Search radius in meters
            
        Returns:
            Dictionary with nearby places information
        """
        try:
            # Get nearby places
            places_result = self.get_nearby_places(latitude, longitude, category, radius)
            
            if not places_result.get("success", False):
                return places_result
            
            places = places_result["places"]
            
            if not places:
                return {
                    "success": True,
                    "message": f"No {category} places found within {radius} meters",
                    "count": 0
                }
            
            # Create a feature group for places
            places_group = folium.FeatureGroup(name=f"Nearby {category}")
            
            # Add places to map
            for place in places:
                # Determine icon
                icon_name = "info"
                if category == "amenity":
                    if place["type"] in ["restaurant", "cafe", "bar", "fast_food"]:
                        icon_name = "cutlery"
                    elif place["type"] in ["hospital", "pharmacy", "doctors"]:
                        icon_name = "plus"
                    elif place["type"] in ["school", "university", "library"]:
                        icon_name = "book"
                    elif place["type"] in ["bank", "atm"]:
                        icon_name = "usd"
                    elif place["type"] in ["parking", "fuel"]:
                        icon_name = "car"
                elif category == "shop":
                    icon_name = "shopping-cart"
                elif category == "tourism":
                    icon_name = "camera"
                
                # Create marker
                marker = folium.Marker(
                    location=(place["latitude"], place["longitude"]),
                    popup=f"{place['name']} ({place['type']})",
                    icon=folium.Icon(icon=icon_name)
                )
                
                # Add marker to group
                marker.add_to(places_group)
            
            # Add group to map
            places_group.add_to(map_object)
            
            logger.info(f"Added {len(places)} {category} places to map")
            return {
                "success": True,
                "count": len(places),
                "category": category
            }
            
        except Exception as e:
            logger.error(f"Add places to map error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_isochrone(self, latitude: float, longitude: float, 
                     mode: str = "walking", time_limits: List[int] = [5, 10, 15]) -> Dict[str, Any]:
        """
        Get isochrone (travel time) polygons.
        
        Args:
            latitude: Latitude
            longitude: Longitude
            mode: Transportation mode ('walking', 'cycling', 'driving')
            time_limits: List of time limits in minutes
            
        Returns:
            Dictionary with isochrone information
        """
        try:
            # Map mode to OSMnx network type
            if mode == "walking":
                network_type = "walk"
            elif mode == "cycling":
                network_type = "bike"
            elif mode == "driving":
                network_type = "drive"
            else:
                network_type = "walk"  # Default
            
            # Get graph
            G = ox.graph_from_point((latitude, longitude), dist=5000, network_type=network_type)
            
            # Get nearest node
            center_node = ox.distance.nearest_nodes(G, longitude, latitude)
            
            # Calculate isochrones
            isochrones = []
            
            for minutes in time_limits:
                # Convert minutes to meters based on mode
                if mode == "walking":
                    # Assume walking speed of 5 km/h
                    distance = (5 * 1000 / 60) * minutes  # meters
                elif mode == "cycling":
                    # Assume cycling speed of 15 km/h
                    distance = (15 * 1000 / 60) * minutes  # meters
                elif mode == "driving":
                    # Assume driving speed of 50 km/h
                    distance = (50 * 1000 / 60) * minutes  # meters
                else:
                    distance = (5 * 1000 / 60) * minutes  # meters
                
                # Get subgraph within distance
                subgraph = nx.ego_graph(G, center_node, radius=distance, distance='length')
                
                # Get node points
                node_points = []
                for node in subgraph.nodes():
                    point = (G.nodes[node]['y'], G.nodes[node]['x'])
                    node_points.append(point)
                
                if node_points:
                    # Create convex hull
                    from scipy.spatial import ConvexHull
                    if len(node_points) >= 3:
                        try:
                            hull = ConvexHull(node_points)
                            hull_points = [node_points[i] for i in hull.vertices]
                        except:
                            hull_points = node_points
                    else:
                        hull_points = node_points
                    
                    isochrones.append({
                        "time_minutes": minutes,
                        "distance_meters": distance,
                        "polygon": hull_points
                    })
            
            logger.info(f"Calculated isochrones for {mode} mode")
            return {
                "success": True,
                "latitude": latitude,
                "longitude": longitude,
                "mode": mode,
                "isochrones": isochrones
            }
            
        except Exception as e:
            logger.error(f"Isochrone error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_isochrones_to_map(self, map_object, isochrone_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add isochrones to a map.
        
        Args:
            map_object: Folium map object
            isochrone_result: Result from get_isochrone
            
        Returns:
            Dictionary with isochrone mapping result
        """
        try:
            if not isochrone_result.get("success", False):
                return {
                    "success": False,
                    "error": "Invalid isochrone result"
                }
            
            # Get isochrones
            isochrones = isochrone_result["isochrones"]
            mode = isochrone_result["mode"]
            
            # Create a feature group for isochrones
            isochrone_group = folium.FeatureGroup(name=f"{mode.capitalize()} Isochrones")
            
            # Add isochrones to map
            colors = ['blue', 'green', 'yellow', 'orange', 'red']
            
            for i, isochrone in enumerate(isochrones):
                # Get polygon
                polygon = isochrone["polygon"]
                
                # Get color
                color = colors[i % len(colors)]
                
                # Create polygon
                folium.Polygon(
                    locations=polygon,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.2,
                    weight=2,
                    popup=f"{isochrone['time_minutes']} minutes"
                ).add_to(isochrone_group)
            
            # Add center marker
            folium.Marker(
                location=(isochrone_result["latitude"], isochrone_result["longitude"]),
                popup="Center",
                icon=folium.Icon(icon="home")
            ).add_to(isochrone_group)
            
            # Add group to map
            isochrone_group.add_to(map_object)
            
            logger.info(f"Added {len(isochrones)} isochrones to map")
            return {
                "success": True,
                "count": len(isochrones),
                "mode": mode
            }
            
        except Exception as e:
            logger.error(f"Add isochrones to map error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_elevation_profile(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get elevation profile for a route.
        
        Args:
            route: Route dictionary from get_route
            
        Returns:
            Dictionary with elevation profile
        """
        try:
            if not route.get("success", False) or not route.get("routes"):
                return {
                    "success": False,
                    "error": "Invalid route"
                }
            
            # Get main route
            main_route = route["routes"][0]
            route_coords = main_route["coordinates"]
            
            # Sample points along the route (max 100 points)
            sample_size = min(len(route_coords), 100)
            indices = np.linspace(0, len(route_coords) - 1, sample_size, dtype=int)
            sampled_coords = [route_coords[i] for i in indices]
            
            # Get elevation data using Open-Elevation API
            elevations = []
            
            # Process in batches of 20 points
            batch_size = 20
            for i in range(0, len(sampled_coords), batch_size):
                batch = sampled_coords[i:i+batch_size]
                
                # Create request
                locations = [{"latitude": lat, "longitude": lon} for lat, lon in batch]
                payload = {"locations": locations}
                
                # Make request
                response = requests.post("https://api.open-elevation.com/api/v1/lookup", json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Extract elevations
                for result in data["results"]:
                    elevations.append({
                        "latitude": result["latitude"],
                        "longitude": result["longitude"],
                        "elevation": result["elevation"]
                    })
            
            # Calculate distances
            distances = [0]
            for i in range(1, len(sampled_coords)):
                prev_coord = sampled_coords[i-1]
                curr_coord = sampled_coords[i]
                distance = geodesic(prev_coord, curr_coord).meters
                distances.append(distances[-1] + distance)
            
            # Create elevation profile
            elevation_profile = []
            for i in range(len(elevations)):
                elevation_profile.append({
                    "distance_meters": distances[i],
                    "elevation_meters": elevations[i]["elevation"],
                    "latitude": elevations[i]["latitude"],
                    "longitude": elevations[i]["longitude"]
                })
            
            logger.info(f"Created elevation profile with {len(elevation_profile)} points")
            return {
                "success": True,
                "profile": elevation_profile,
                "min_elevation": min(p["elevation_meters"] for p in elevation_profile),
                "max_elevation": max(p["elevation_meters"] for p in elevation_profile),
                "total_ascent": sum(max(0, elevation_profile[i]["elevation_meters"] - elevation_profile[i-1]["elevation_meters"]) 
                                  for i in range(1, len(elevation_profile))),
                "total_descent": sum(max(0, elevation_profile[i-1]["elevation_meters"] - elevation_profile[i]["elevation_meters"]) 
                                   for i in range(1, len(elevation_profile)))
            }
            
        except Exception as e:
            logger.error(f"Elevation profile error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def plot_elevation_profile(self, elevation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plot elevation profile.
        
        Args:
            elevation_result: Result from get_elevation_profile
            
        Returns:
            Dictionary with plot information
        """
        try:
            if not elevation_result.get("success", False):
                return {
                    "success": False,
                    "error": "Invalid elevation profile"
                }
            
            # Get profile data
            profile = elevation_result["profile"]
            
            # Extract data
            distances = [p["distance_meters"] / 1000 for p in profile]  # Convert to km
            elevations = [p["elevation_meters"] for p in profile]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(distances, elevations)
            plt.fill_between(distances, elevations, min(elevations), alpha=0.3)
            
            # Add labels and title
            plt.xlabel('Distance (km)')
            plt.ylabel('Elevation (m)')
            plt.title('Elevation Profile')
            
            # Add grid
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add elevation stats
            min_elev = elevation_result["min_elevation"]
            max_elev = elevation_result["max_elevation"]
            total_ascent = elevation_result["total_ascent"]
            total_descent = elevation_result["total_descent"]
            
            plt.figtext(0.02, 0.02, 
                       f"Min: {min_elev:.1f}m, Max: {max_elev:.1f}m\nAscent: {total_ascent:.1f}m, Descent: {total_descent:.1f}m",
                       bbox=dict(facecolor='white', alpha=0.8))
            
            # Save plot to file
            plot_file = os.path.join(self.cache_dir, f"elevation_profile_{int(time.time())}.png")
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created elevation profile plot: {plot_file}")
            return {
                "success": True,
                "plot_file": plot_file,
                "min_elevation": min_elev,
                "max_elevation": max_elev,
                "total_ascent": total_ascent,
                "total_descent": total_descent
            }
            
        except Exception as e:
            logger.error(f"Elevation plot error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_navigation_map(self, origin: str, destination: str, 
                            mode: str = "driving") -> Dict[str, Any]:
        """
        Create a complete navigation map with route.
        
        Args:
            origin: Origin location string
            destination: Destination location string
            mode: Transportation mode
            
        Returns:
            Dictionary with navigation map information
        """
        try:
            # Get route
            route_result = self.get_route(origin, destination, mode, alternatives=True)
            
            if not route_result.get("success", False):
                return route_result
            
            # Create map centered between origin and destination
            origin_coords = route_result["origin_coords"]
            dest_coords = route_result["destination_coords"]
            
            center_lat = (origin_coords[0] + dest_coords[0]) / 2
            center_lng = (origin_coords[1] + dest_coords[1]) / 2
            
            map_result = self.create_map((center_lat, center_lng), zoom=12)
            
            if not map_result.get("success", False):
                return map_result
            
            map_object = map_result["map_object"]
            
            # Add route to map
            route_map_result = self.add_route_to_map(map_object, route_result)
            
            if not route_map_result.get("success", False):
                return route_map_result
            
            # Get nearby places at origin and destination
            self.add_nearby_places_to_map(map_object, origin_coords[0], origin_coords[1], "amenity", 500)
            self.add_nearby_places_to_map(map_object, dest_coords[0], dest_coords[1], "amenity", 500)
            
            # Save map
            save_result = self.save_map(map_object)
            
            if not save_result.get("success", False):
                return save_result
            
            # Get elevation profile
            elevation_result = self.get_elevation_profile(route_result)
            
            if elevation_result.get("success", False):
                # Plot elevation profile
                plot_result = self.plot_elevation_profile(elevation_result)
                elevation_plot = plot_result.get("plot_file") if plot_result.get("success", False) else None
            else:
                elevation_plot = None
            
            logger.info(f"Created navigation map from {origin} to {destination}")
            return {
                "success": True,
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "map_file": save_result["file_path"],
                "elevation_plot": elevation_plot,
                "route": route_result
            }
            
        except Exception as e:
            logger.error(f"Navigation map error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_traffic_data(self, latitude: float, longitude: float, radius: int = 5000) -> Dict[str, Any]:
        """
        Get traffic data for an area.
        
        Args:
            latitude: Latitude
            longitude: Longitude
            radius: Search radius in meters
            
        Returns:
            Dictionary with traffic data
        """
        try:
            # Get road network
            G = ox.graph_from_point((latitude, longitude), dist=radius, network_type='drive')
            
            # Get edge centrality as a proxy for traffic importance
            edge_centrality = nx.edge_betweenness_centrality(G)
            
            # Normalize centrality values
            max_centrality = max(edge_centrality.values())
            normalized_centrality = {k: v / max_centrality for k, v in edge_centrality.items()}
            
            # Add centrality to graph
            for u, v, k, data in G.edges(keys=True, data=True):
                data['centrality'] = normalized_centrality.get((u, v), 0)
            
            # Create GeoDataFrame from graph
            gdf_edges = ox.graph_to_gdfs(G, nodes=False)
            
            # Process edges
            roads = []
            for idx, row in gdf_edges.iterrows():
                u, v, k = idx
                
                # Get road name
                name = row.get('name', 'Unnamed Road')
                if isinstance(name, list):
                    name = name[0] if name else 'Unnamed Road'
                
                # Get road type
                road_type = row.get('highway', 'road')
                if isinstance(road_type, list):
                    road_type = road_type[0] if road_type else 'road'
                
                # Get centrality
                centrality = row.get('centrality', 0)
                
                # Get coordinates
                if 'geometry' in row:
                    geom = row['geometry']
                    if hasattr(geom, 'coords'):
                        coords = list(geom.coords)
                    else:
                        continue
                else:
                    continue
                
                roads.append({
                    "name": name,
                    "type": road_type,
                    "centrality": centrality,
                    "coordinates": [(y, x) for x, y in coords],
                    "length": row.get('length', 0)
                })
            
            logger.info(f"Got traffic data for {len(roads)} roads")
            return {
                "success": True,
                "latitude": latitude,
                "longitude": longitude,
                "radius": radius,
                "roads": roads,
                "road_count": len(roads)
            }
            
        except Exception as e:
            logger.error(f"Traffic data error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_traffic_to_map(self, map_object, traffic_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add traffic data to a map.
        
        Args:
            map_object: Folium map object
            traffic_result: Result from get_traffic_data
            
        Returns:
            Dictionary with traffic mapping result
        """
        try:
            if not traffic_result.get("success", False):
                return {
                    "success": False,
                    "error": "Invalid traffic data"
                }
            
            # Get roads
            roads = traffic_result["roads"]
            
            # Create a feature group for traffic
            traffic_group = folium.FeatureGroup(name="Traffic")
            
            # Add roads to map
            for road in roads:
                # Determine color based on centrality (traffic importance)
                centrality = road["centrality"]
                
                if centrality > 0.8:
                    color = 'red'
                elif centrality > 0.6:
                    color = 'orange'
                elif centrality > 0.4:
                    color = 'yellow'
                elif centrality > 0.2:
                    color = 'blue'
                else:
                    color = 'green'
                
                # Create line
                folium.PolyLine(
                    locations=road["coordinates"],
                    color=color,
                    weight=3 + centrality * 5,  # Width based on centrality
                    opacity=0.7,
                    popup=f"{road['name']} ({road['type']})"
                ).add_to(traffic_group)
            
            # Add group to map
            traffic_group.add_to(map_object)
            
            logger.info(f"Added traffic data for {len(roads)} roads to map")
            return {
                "success": True,
                "road_count": len(roads)
            }
            
        except Exception as e:
            logger.error(f"Add traffic to map error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_directions(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get turn-by-turn directions for a route.
        
        Args:
            route: Route dictionary from get_route
            
        Returns:
            Dictionary with directions
        """
        try:
            if not route.get("success", False) or not route.get("routes"):
                return {
                    "success": False,
                    "error": "Invalid route"
                }
            
            # Use OSRM API for directions
            origin_coords = route["origin_coords"]
            dest_coords = route["destination_coords"]
            mode = route["mode"]
            
            # Map mode to OSRM profile
            if mode == "driving":
                profile = "car"
            elif mode == "walking":
                profile = "foot"
            elif mode == "cycling":
                profile = "bike"
            else:
                profile = "car"  # Default
            
            # Build OSRM API URL
            base_url = "https://router.project-osrm.org/route/v1"
            coords = f"{origin_coords[1]},{origin_coords[0]};{dest_coords[1]},{dest_coords[0]}"
            options = "steps=true&annotations=true&geometries=geojson&overview=full"
            url = f"{base_url}/{profile}/{coords}?{options}"
            
            # Make request
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data["code"] != "Ok":
                return {
                    "success": False,
                    "error": f"OSRM API error: {data['code']}"
                }
            
            # Process directions
            route_data = data["routes"][0]
            legs = route_data["legs"]
            
            directions = []
            for leg in legs:
                for step in leg["steps"]:
                    # Get step details
                    maneuver = step["maneuver"]
                    instruction = maneuver["type"]
                    
                    if "modifier" in maneuver:
                        instruction += f" {maneuver['modifier']}"
                    
                    # Get distance and duration
                    distance = step["distance"]  # meters
                    duration = step["duration"]  # seconds
                    
                    # Get start location
                    location = maneuver["location"]
                    longitude, latitude = location
                    
                    directions.append({
                        "instruction": step["name"] if step["name"] else instruction,
                        "distance_meters": distance,
                        "duration_seconds": duration,
                        "latitude": latitude,
                        "longitude": longitude
                    })
            
            logger.info(f"Got {len(directions)} direction steps")
            return {
                "success": True,
                "directions": directions,
                "total_distance": route_data["distance"],
                "total_duration": route_data["duration"]
            }
            
        except Exception as e:
            logger.error(f"Directions error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_directions_map(self, origin: str, destination: str, 
                            mode: str = "driving") -> Dict[str, Any]:
        """
        Create a map with turn-by-turn directions.
        
        Args:
            origin: Origin location string
            destination: Destination location string
            mode: Transportation mode
            
        Returns:
            Dictionary with directions map information
        """
        try:
            # Get route
            route_result = self.get_route(origin, destination, mode)
            
            if not route_result.get("success", False):
                return route_result
            
            # Get directions
            directions_result = self.get_directions(route_result)
            
            if not directions_result.get("success", False):
                return directions_result
            
            # Create map
            map_result = self.create_navigation_map(origin, destination, mode)
            
            if not map_result.get("success", False):
                return map_result
            
            # Add directions to map HTML
            map_file = map_result["map_file"]
            
            # Read map HTML
            with open(map_file, 'r') as f:
                map_html = f.read()
            
            # Create directions HTML
            directions_html = """
            <div style="position: absolute; top: 10px; right: 10px; width: 300px; max-height: 80%; overflow-y: auto; background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.3); z-index: 1000;">
                <h3>Directions</h3>
                <p><strong>From:</strong> {origin}</p>
                <p><strong>To:</strong> {destination}</p>
                <p><strong>Distance:</strong> {distance:.1f} km</p>
                <p><strong>Duration:</strong> {duration:.1f} min</p>
                <hr>
                <ol>
            """.format(
                origin=route_result["origin"],
                destination=route_result["destination"],
                distance=directions_result["total_distance"] / 1000,
                duration=directions_result["total_duration"] / 60
            )
            
            for i, step in enumerate(directions_result["directions"]):
                directions_html += f"""
                <li style="margin-bottom: 10px;">
                    <strong>{step['instruction']}</strong><br>
                    {step['distance_meters']:.0f} m ({step['duration_seconds']:.0f} sec)
                </li>
                """
            
            directions_html += """
                </ol>
            </div>
            """
            
            # Insert directions HTML before </body>
            map_html = map_html.replace('</body>', f'{directions_html}</body>')
            
            # Write updated HTML
            with open(map_file, 'w') as f:
                f.write(map_html)
            
            logger.info(f"Created directions map from {origin} to {destination}")
            return {
                "success": True,
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "map_file": map_file,
                "directions": directions_result["directions"],
                "direction_count": len(directions_result["directions"])
            }
            
        except Exception as e:
            logger.error(f"Directions map error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Example usage
if __name__ == "__main__":
    # Create navigation tool
    nav_tool = NavigationTool()
    
    # Geocode a location
    location = "New York, NY"
    geocode_result = nav_tool.geocode(location)
    print(f"Geocode result: {geocode_result}")
    
    # Get a route
    origin = "New York, NY"
    destination = "Boston, MA"
    route_result = nav_tool.get_route(origin, destination, "driving")
    print(f"Route result: {route_result}")
    
    # Create a navigation map
    map_result = nav_tool.create_navigation_map(origin, destination, "driving")
    print(f"Map result: {map_result}")
    
    # Open the map
    if map_result.get("success", False):
        nav_tool.open_map(map_result["map_file"])