"""
Global Configuration and Constants for Ghost Hunter.
Elegantly restyled with Grey, Silver, Slate, and Gold/Yellow accents.
Uses standard serif typography (Georgia / Times New Roman) with anti-aliasing.
"""

# Grid & Dynamics Parameters (Defaults)
GRID_WIDTH: int = 10
GRID_HEIGHT: int = 10
TURN_LIMIT_T_MAX: int = 100

# Difficulty Modes mapping
DIFFICULTY_CONFIGS: dict[str, dict] = {
    "Beginner": {
        "width": 10,
        "height": 10,
        "cell_size": 58,
        "turn_limit": 100,
        "density": 0.16,
        "description": "10x10 Grid · Perfect for quick tactical learning"
    },
    "Intermediate": {
        "width": 15,
        "height": 15,
        "cell_size": 40,
        "turn_limit": 150,
        "density": 0.18,
        "description": "15x15 Grid · Balanced maze with increased uncertainty"
    },
    "Hard": {
        "width": 20,
        "height": 20,
        "cell_size": 30,
        "turn_limit": 200,
        "density": 0.20,
        "description": "20x20 Grid · Dense labyrinth, demanding precision"
    },
    "Extreme": {
        "width": 30,
        "height": 30,
        "cell_size": 20,
        "turn_limit": 300,
        "density": 0.22,
        "description": "30x30 Grid · Massive scale challenge with sparse pings"
    }
}

# A* Path Hint Ability Parameters
ASTAR_HINT_DURATION_SEC: float = 3.0    # Path visibly glowing for 2.5 - 3.0 seconds
ASTAR_HINT_COOLDOWN_SEC: float = 5.0    # 5.0 seconds cooldown before recharge
GHOST_MOVE_INTERVAL_SEC: float = 1.0    # Ghost moves every 1.0 second in real-time

# Sensor & Probability Defaults
DEFAULT_SENSOR_SIGMA: float = 1.2
PROBABILITY_FLOOR_EPSILON: float = 1e-7

# Benchmark & Experiment Setup
EXPERIMENT_SEEDS: list[int] = [42, 101, 202, 303, 404]
NOISE_SIGMA_LEVELS: list[float] = [0.5, 1.0, 1.5, 2.0, 3.0]
NUM_BENCHMARK_TRIALS: int = 100
NUM_MAZE_TOPOLOGIES: int = 10

# Movement Actions
ACTION_UP: tuple[int, int] = (-1, 0)
ACTION_DOWN: tuple[int, int] = (1, 0)
ACTION_LEFT: tuple[int, int] = (0, -1)
ACTION_RIGHT: tuple[int, int] = (0, 1)
ACTION_STAY: tuple[int, int] = (0, 0)

MOVES_DICT: dict[str, tuple[int, int]] = {
    "UP": ACTION_UP,
    "DOWN": ACTION_DOWN,
    "LEFT": ACTION_LEFT,
    "RIGHT": ACTION_RIGHT,
    "STAY": ACTION_STAY
}

GHOST_ACTIONS: list[tuple[int, int]] = [
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY
]

# -----------------------------------------------------------------------------
# Color Contrast: Silver, Slate Grey, Charcoal & Gold/Yellow Accents
# -----------------------------------------------------------------------------
# Outer Canvas & Window
COLOR_WINDOW_BG: tuple[int, int, int] = (248, 250, 252)       # Clean Soft Silver/Platinum

# Grid Canvas Area
COLOR_GRID_OUTER_FRAME: tuple[int, int, int] = (203, 213, 225) # Silver/Chrome Border
COLOR_GRID_BG: tuple[int, int, int] = (15, 23, 42)             # Deep Charcoal-Slate Interior
COLOR_GRID_SURFACE: tuple[int, int, int] = (15, 23, 42)        # Empty cell background
COLOR_GRID_LINE: tuple[int, int, int] = (30, 41, 59)           # Subtle cell border line

# Distinct Obstacle Walls
COLOR_WALL: tuple[int, int, int] = (71, 85, 105)               # Slate-600 Silver Stone Wall
COLOR_WALL_INNER: tuple[int, int, int] = (51, 65, 85)          # Slate-700 Core Plate
COLOR_WALL_ACCENT: tuple[int, int, int] = (226, 232, 240)      # High-contrast Silver Border
COLOR_WALL_RIVET: tuple[int, int, int] = (234, 179, 8)         # Gold rivet point

# Entities
COLOR_HUNTER: tuple[int, int, int] = (14, 165, 233)            # Radiant Sky/Cyan Hunter
COLOR_HUNTER_RING: tuple[int, int, int] = (186, 230, 253)      # Silver-Cyan Aura
COLOR_GHOST: tuple[int, int, int] = (239, 68, 68)             # Crimson / Rose Ghost
COLOR_GHOST_AURA: tuple[int, int, int] = (254, 202, 202)
COLOR_TARGET_MARKER: tuple[int, int, int] = (250, 204, 21)     # Luminous Gold Crosshair Reticle (Yellow Pointer)

# A* Path Glow
COLOR_ASTAR_PATH: tuple[int, int, int] = (234, 179, 8)         # Vivid Amber Gold Path
COLOR_ASTAR_PATH_PULSE: tuple[int, int, int] = (254, 240, 138) # Soft Yellow Glow

# Sidebar Cards
CARD1_BG: tuple[int, int, int] = (241, 245, 249)               # Soft Silver Card
CARD1_BORDER: tuple[int, int, int] = (203, 213, 225)           # Crisp Silver Border
CARD1_TITLE: tuple[int, int, int] = (30, 41, 59)              # Deep Charcoal Header
TAG_PILL_BG: tuple[int, int, int] = (202, 138, 4)              # Gold Pill
TAG_PILL_TEXT: tuple[int, int, int] = (255, 255, 255)
STATUS_ACTIVE_BG: tuple[int, int, int] = (254, 240, 138)       # Soft Gold Highlight
STATUS_ACTIVE_TEXT: tuple[int, int, int] = (133, 77, 14)
BTN_RESTART_BG: tuple[int, int, int] = (51, 65, 85)            # Slate Button
BTN_RESTART_HOVER: tuple[int, int, int] = (71, 85, 105)

# Card 2: Mission Telemetry
CARD2_BG: tuple[int, int, int] = (248, 250, 252)
CARD2_BORDER: tuple[int, int, int] = (226, 232, 240)
CARD2_TITLE: tuple[int, int, int] = (30, 41, 59)
CARD2_LABEL: tuple[int, int, int] = (100, 116, 139)
CARD2_VALUE: tuple[int, int, int] = (15, 23, 42)

# Card 3: HMM Belief State
CARD3_BG: tuple[int, int, int] = (254, 252, 232)
CARD3_BORDER: tuple[int, int, int] = (254, 240, 138)
CARD3_TITLE: tuple[int, int, int] = (161, 98, 7)
CARD3_LABEL: tuple[int, int, int] = (113, 63, 18)
CARD3_VALUE: tuple[int, int, int] = (26, 46, 5)

# Card 4: Keyboard Shortcuts & Controls
CARD4_BG: tuple[int, int, int] = (241, 245, 249)
CARD4_BORDER: tuple[int, int, int] = (203, 213, 225)
CARD4_TITLE: tuple[int, int, int] = (30, 41, 59)
CARD4_KEY: tuple[int, int, int] = (180, 83, 9)
CARD4_DESC: tuple[int, int, int] = (71, 85, 105)

# Buttons in Menus
BTN_PRIMARY_BG: tuple[int, int, int] = (202, 138, 4)           # Vibrant Gold Button
BTN_PRIMARY_HOVER: tuple[int, int, int] = (234, 179, 8)        # Light Gold Hover
BTN_SECONDARY_BG: tuple[int, int, int] = (71, 85, 105)         # Silver-Slate Button
BTN_SECONDARY_HOVER: tuple[int, int, int] = (100, 116, 139)
BTN_DANGER_BG: tuple[int, int, int] = (185, 28, 28)            # Crimson Red Exit
BTN_DANGER_HOVER: tuple[int, int, int] = (220, 38, 38)

# Heatmap Colors
HEATMAP_P0: tuple[int, int, int] = (15, 23, 42)
HEATMAP_P1: tuple[int, int, int] = (30, 58, 138)              # Deep Blue
HEATMAP_P2: tuple[int, int, int] = (234, 179, 8)              # Luminous Gold
HEATMAP_P3: tuple[int, int, int] = (239, 68, 68)              # Vivid Coral Pink
