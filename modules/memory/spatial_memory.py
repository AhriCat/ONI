class SpatialMemoryModule:
    def __init__(self, room_size: Tuple[int, int], overlap: float = 0.2, max_memory: int = 100):
        """
        Initializes the spatial memory with parameters defining room size, overlap, and memory constraints.

        Args:
            room_size (tuple): Dimensions of each room (width, height).
            overlap (float): Fractional overlap between adjacent rooms.
            max_memory (int): Maximum number of rooms to retain in memory.
        """
        self.room_width, self.room_height = room_size
        self.overlap = overlap
        self.current_position = (0, 0)  # Starting at origin
        self.memory = {}  # Dictionary to store room data indexed by position
        self.max_memory = max_memory

    def get_current_room_key(self) -> Tuple[int, int]:
        """
        Determines the key for the current room based on Oni's position.

        Returns:
            tuple: Coordinates representing the current room.
        """
        x, y = self.current_position
        room_x = int(x // (self.room_width * (1 - self.overlap)))
        room_y = int(y // (self.room_height * (1 - self.overlap)))
        return (room_x, room_y)

    def update_position(self, new_position: Tuple[int, int]) -> bool:
        """
        Updates Oni's current position and determines if a new room needs to be loaded.

        Args:
            new_position (tuple): New (x, y) coordinates.

        Returns:
            bool: True if a new room is entered, False otherwise.
        """
        old_room = self.get_current_room_key()
        self.current_position = new_position
        new_room = self.get_current_room_key()
        if new_room != old_room:
            return True
        return False

    def load_room(self, room_key: Tuple[int, int], room_data: Dict):
        """
        Loads data for a new room into memory.

        Args:
            room_key (tuple): Coordinates representing the room.
            room_data (dict): Data associated with the room.
        """
        self.memory[room_key] = room_data
        # If memory exceeds max_memory, remove the least recently used room
        if len(self.memory) > self.max_memory:
            oldest_room = next(iter(self.memory))
            del self.memory[oldest_room]

    def get_current_room_data(self) -> Optional[Dict]:
        """
        Retrieves data for the current room.

        Returns:
            dict or None: Data of the current room or None if not loaded.
        """
        room_key = self.get_current_room_key()
        return self.memory.get(room_key, None)
