import pyautogui
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
from ExecutiveFunction import exec_func
@dataclass
class ScreenRegion:
    """Defines a named region on the screen with coordinates"""
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    def center(self) -> Tuple[int, int]:
        """Get center coordinates of the region"""
        return ((self.x2 + self.x1) // 2, (self.y2 + self.y1) // 2)

    def contains(self, x: int, y: int) -> bool:
        """Check if coordinates are within the region"""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

class DisplayMap:
    """Manages screen regions and coordinates"""
    def __init__(self):
        self.regions: Dict[str, ScreenRegion] = {}
        self.screen_width, self.screen_height = pyautogui.size()

    def add_region(self, name: str, x1: int, y1: int, x2: int, y2: int):
        """Add a named region to the map"""
        self.regions[name] = ScreenRegion(name, x1, y1, x2, y2)

    def get_region(self, name: str) -> Optional[ScreenRegion]:
        """Get a region by name"""
        return self.regions.get(name)

class ActionSequence:
    """Defines a sequence of actions that can be executed together"""
    def __init__(self, name: str, actions: List[dict]):
        self.name = name
        self.actions = actions

    def execute(self, controller: 'KeyboardMouseController'):
        """Execute all actions in sequence"""
        for action in self.actions:
            controller.execute_action(action)
            time.sleep(0.01)  # Small delay between actions

class KeyboardMouseController:
    def __init__(self):
        self.pyautogui = pyautogui
        self.display_map = DisplayMap()
        self.action_sequences: Dict[str, ActionSequence] = {}
        self.initialize_safe_settings()
        self.initialize_common_regions()
        self.initialize_common_sequences()

    def initialize_safe_settings(self):
        """Initialize safety settings for PyAutoGUI"""
        self.pyautogui.FAILSAFE = True
        self.pyautogui.PAUSE = 0.0  # No pause between actions for faster response

    def initialize_common_regions(self):
        """Set up common screen regions"""
        # Get screen dimensions
        width, height = self.pyautogui.size()

        # Define common regions
        self.display_map.add_region("taskbar", 0, height - 40, width, height)
        self.display_map.add_region("top_bar", 0, 0, width, 40)

        # Split screen into 9 sections for general navigation
        section_w = width // 3
        section_h = height // 3

        for i in range(3):
            for j in range(3):
                region_name = f"section_{i}_{j}"
                self.display_map.add_region(
                    region_name,
                    i * section_w,
                    j * section_h,
                    (i + 1) * section_w,
                    (j + 1) * section_h
                )

    def initialize_common_sequences(self):
        """Initialize common action sequences"""
        # Copy text sequence
        self.add_sequence("copy_text", [
            {"type": "key_down", "key": "ctrl"},
            {"type": "press_key", "key": "c"},
            {"type": "key_up", "key": "ctrl"}
        ])

        # Paste text sequence
        self.add_sequence("paste_text", [
            {"type": "key_down", "key": "ctrl"},
            {"type": "press_key", "key": "v"},
            {"type": "key_up", "key": "ctrl"}
        ])

        # Save file sequence
        self.add_sequence("save_file", [
            {"type": "key_down", "key": "ctrl"},
            {"type": "press_key", "key": "s"},
            {"type": "key_up", "key": "ctrl"}
        ])

        # Game action sequences
        self.add_sequence("game_move_forward", [
            {"type": "key_down", "key": "w"}
        ])
        self.add_sequence("game_stop_forward", [
            {"type": "key_up", "key": "w"}
        ])
        self.add_sequence("game_move_left", [
            {"type": "key_down", "key": "a"}
        ])
        self.add_sequence("game_stop_left", [
            {"type": "key_up", "key": "a"}
        ])
        self.add_sequence("game_move_backward", [
            {"type": "key_down", "key": "s"}
        ])
        self.add_sequence("game_stop_backward", [
            {"type": "key_up", "key": "s"}
        ])
        self.add_sequence("game_move_right", [
            {"type": "key_down", "key": "d"}
        ])
        self.add_sequence("game_stop_right", [
            {"type": "key_up", "key": "d"}
        ])

    def add_sequence(self, name: str, actions: List[dict]):
        """Add a new action sequence"""
        self.action_sequences[name] = ActionSequence(name, actions)

    def execute_sequence(self, sequence_name: str):
        """Execute a named action sequence"""
        if sequence_name in self.action_sequences:
            self.action_sequences[sequence_name].execute(self)
        else:
            raise ValueError(f"Unknown sequence: {sequence_name}")

    def move_to_region(self, region_name: str, duration: float = 0.0):
        """Move mouse to the center of a named region"""
        region = self.display_map.get_region(region_name)
        if region:
            center_x, center_y = region.center()
            self.move_mouse(center_x, center_y, duration)

    def click_in_region(self, region_name: str, button: str = 'left'):
        """Click in the center of a named region"""
        region = self.display_map.get_region(region_name)
        if region:
            center_x, center_y = region.center()
            self.click_mouse(center_x, center_y, button)

    def execute_action(self, action: dict):
        """Execute an action based on the action dictionary"""
        action_type = action.get('type')
        if action_type == 'press_key':
            key = action.get('key')
            self.press_key(key)
        elif action_type == 'key_down':
            key = action.get('key')
            self.key_down(key)
        elif action_type == 'key_up':
            key = action.get('key')
            self.key_up(key)
        elif action_type == 'typewrite':
            message = action.get('message', '')
            interval = action.get('interval', 0.0)
            self.typewrite(message, interval)
        elif action_type == 'move_mouse':
            x = action.get('x')
            y = action.get('y')
            duration = action.get('duration', 0.0)
            if x is not None and y is not None:
                self.move_mouse(x, y, duration)
        elif action_type == 'move_mouse_relative':
            x_offset = action.get('x_offset', 0)
            y_offset = action.get('y_offset', 0)
            duration = action.get('duration', 0.0)
            self.move_mouse_relative(x_offset, y_offset, duration)
        elif action_type == 'click_mouse':
            x = action.get('x')
            y = action.get('y')
            button = action.get('button', 'left')
            if x is not None and y is not None:
                self.click_mouse(x, y, button)
            else:
                self.click_mouse(button=button)
        elif action_type == 'scroll':
            clicks = action.get('clicks', 0)
            x = action.get('x')
            y = action.get('y')
            self.scroll(clicks, x, y)
        elif action_type == 'hotkey':
            keys = action.get('keys', [])
            self.hotkey(*keys)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def press_key(self, key: str):
        """Press a key"""
        self.pyautogui.press(key)

    def typewrite(self, message: str, interval: float = 0.0):
        """Type a message with customizable interval"""
        self.pyautogui.typewrite(message, interval=interval)

    def key_down(self, key: str):
        """Press and hold a key"""
        self.pyautogui.keyDown(key)

    def key_up(self, key: str):
        """Release a key"""
        self.pyautogui.keyUp(key)

    def move_mouse(self, x: int, y: int, duration: float = 0.0):
        """Move mouse to a specific position"""
        self.pyautogui.moveTo(x, y, duration)

    def move_mouse_relative(self, x_offset: int, y_offset: int, duration: float = 0.0):
        """Move mouse relative to its current position"""
        self.pyautogui.moveRel(x_offset, y_offset, duration)

    def click_mouse(self, x: Optional[int] = None, y: Optional[int] = None, button: str = 'left'):
        """Click at the specified position"""
        if x is not None and y is not None:
            self.pyautogui.click(x, y, button=button)
        else:
            self.pyautogui.click(button=button)

    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None):
        """Scroll the mouse wheel"""
        if x is not None and y is not None:
            self.pyautogui.scroll(clicks, x, y)
        else:
            self.pyautogui.scroll(clicks)

    def hotkey(self, *keys):
        """Press multiple keys in sequence"""
        self.pyautogui.hotkey(*keys)

    def get_pixel_color(self, x: int, y: int) -> Tuple[int, int, int]:
        """Get the RGB color of a pixel at the specified coordinates"""
        screenshot = self.pyautogui.screenshot(region=(x, y, 1, 1))
        return screenshot.getpixel((0, 0))

    # Additional methods for game controls
    def perform_game_action(self, action: dict):
        """Perform an action suitable for game playing"""
        action_type = action.get('type')
        if action_type == 'move':
            key = action.get('key')
            state = action.get('state', 'down')  # 'down' or 'up'
            if state == 'down':
                self.key_down(key)
            elif state == 'up':
                self.key_up(key)
        elif action_type == 'look':
            x_offset = action.get('x_offset', 0)
            y_offset = action.get('y_offset', 0)
            self.move_mouse_relative(x_offset, y_offset)
        elif action_type == 'action':
            key = action.get('key')
            self.press_key(key)
        elif action_type == 'mouse_click':
            button = action.get('button', 'left')
            self.click_mouse(button=button)
        else:
            self.execute_action(action)  # For other action types

# Example usage:
def create_controller():
    controller = KeyboardMouseController()

    # Add custom regions
    controller.display_map.add_region("browser_url", 100, 40, 800, 70)
    controller.display_map.add_region("main_content", 50, 100, 950, 700)

    # Add custom action sequences
    controller.add_sequence("open_new_tab", [
        {"type": "key_down", "key": "ctrl"},
        {"type": "press_key", "key": "t"},
        {"type": "key_up", "key": "ctrl"}
    ])

    # Add game action sequences
    controller.add_sequence("jump", [
        {"type": "press_key", "key": "space"}
    ])
    controller.add_sequence("sprint_start", [
        {"type": "key_down", "key": "shift"}
    ])
    controller.add_sequence("sprint_stop", [
        {"type": "key_up", "key": "shift"}
    ])

    return controller
