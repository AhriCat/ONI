class HeuristicManager:
    def __init__(self, heuristic_function, max_priority: int = 100):
        """
        Initializes the Heuristic Manager.

        Args:
            heuristic_function (callable): Function to compute priority based on room key.
            max_priority (int): Maximum number of rooms to prioritize.
        """
        self.heuristic_function = heuristic_function
        self.priority_queue = []
        self.max_priority = max_priority

    def add_room(self, room_key: Tuple[int, int]):
        priority = self.heuristic_function(room_key)
        heapq.heappush(self.priority_queue, (priority, room_key))
        # Ensure the queue doesn't exceed max_priority
        if len(self.priority_queue) > self.max_priority:
            heapq.heappop(self.priority_queue)

    def get_next_room(self) -> Optional[Tuple[int, int]]:
        if self.priority_queue:
            return heapq.heappop(self.priority_queue)[1]
        return None
