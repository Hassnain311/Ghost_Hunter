"""
Global Configuration and Constants for Ghost Hunter.
Horror Theme: Darkness, Blood Red, Eerie Violet, Hellfire Amber, and Ghostly Cyan.
Uses spooky atmospheric typography with fallback fonts.
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
        "ghost_speed": 1.0,         # Moves every 1.0s
        "glimpse_interval": 8.0,     # Glimpse appears every 8 seconds
        "glimpse_duration": 0.8,     # Visible for 0.8s (<= 1s)
        "description": "10x10 Grid · Perfect for learning (Ghost: 1.0s, Unlimited A* Radar)"
    },
    "Intermediate": {
        "width": 15,
        "height": 15,
        "cell_size": 40,
        "turn_limit": 150,
        "density": 0.18,
        "ghost_speed": 0.75,        # Faster: moves every 0.75s
        "glimpse_interval": 8.0,     # Glimpse appears every 8 seconds
        "glimpse_duration": 0.8,     # Visible for 0.8s (<= 1s)
        "description": "15x15 Grid · Swift ghost (Ghost: 0.75s, Unlimited A* Radar)"
    },
    "Hard": {
        "width": 20,
        "height": 20,
        "cell_size": 30,
        "turn_limit": 200,
        "density": 0.20,
        "ghost_speed": 0.55,        # Rapid: moves every 0.55s
        "glimpse_interval": 8.0,     # Glimpse appears every 8 seconds
        "glimpse_duration": 0.7,     # Visible for 0.7s (<= 1s)
        "description": "20x20 Grid · Agile ghost (Ghost: 0.55s, Unlimited A* Radar)"
    },
    "Extreme": {
        "width": 30,
        "height": 30,
        "cell_size": 20,
        "turn_limit": 300,
        "density": 0.22,
        "ghost_speed": 0.40,        # Blazing fast: moves every 0.40s
        "glimpse_interval": 8.0,     # Glimpse appears every 8 seconds
        "glimpse_duration": 0.6,     # Visible for 0.6s (<= 1s)
        "description": "30x30 Grid · Hyper-speed phantom (Ghost: 0.40s, Unlimited A* Radar)"
    }
}

# A* Path Hint Ability Parameters
ASTAR_HINT_DURATION_SEC: float = 3.0    # Path visibly glowing for 2.5 - 3.0 seconds
ASTAR_HINT_COOLDOWN_SEC: float = 5.0    # 5.0 seconds cooldown before recharge
GHOST_MOVE_INTERVAL_SEC: float = 1.0    # Ghost moves every 1.0 second in real-time
HUMAN_MOVE_INTERVAL_SEC: float = 0.16

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
DISPLAY_FONTS = ("georgia", "liberation serif", "times new roman", "arial")
DATA_FONTS = ("consolas", "lucida console", "courier new", "monospace")

# Horror Palette: Pitch Black, Blood Red, Dark Violet & Spectral Glow
# -----------------------------------------------------------------------------
# Outer Canvas & Window
COLOR_WINDOW_BG: tuple[int, int, int] = (5, 5, 10)            # Pitch Obsidian Black

# Grid Canvas Area
COLOR_GRID_OUTER_FRAME: tuple[int, int, int] = (154, 125, 79)  # Aged brass moulding
COLOR_GRID_BG: tuple[int, int, int] = (28, 30, 25)             # Dark metal under the moulding
COLOR_GRID_SURFACE: tuple[int, int, int] = (81, 88, 77)        # Weathered limestone
COLOR_GRID_LINE: tuple[int, int, int] = (42, 46, 39)           # Mortar between flagstones

# Posterior display, from zero mass through spectral teal to the current MAP peak.
BELIEF_LOW = (36, 49, 48)
BELIEF_MID = (48, 151, 143)
BELIEF_PEAK = (255, 219, 134)

# Proximity Heatmap Colors (Manhattan Distance to HMM MAP Estimate)
HEATMAP_DIST_1: tuple[int, int, int] = (225, 67, 84)          # Crimson (Distance 1)
HEATMAP_DIST_2_3: tuple[int, int, int] = (245, 174, 80)       # Candle amber (Distance 2, 3)
HEATMAP_DIST_4_5: tuple[int, int, int] = (57, 191, 146)       # Spectral jade (Distance 4, 5)

# Obstacle Walls (Crypt Cursed Obelisks)
COLOR_WALL: tuple[int, int, int] = (23, 13, 34)               # Cursed Dark Violet Stone
COLOR_WALL_INNER: tuple[int, int, int] = (12, 7, 21)         # Inner Obsidian Core
COLOR_WALL_ACCENT: tuple[int, int, int] = (139, 92, 246)      # Neon Phantom Violet Runes
COLOR_WALL_RIVET: tuple[int, int, int] = (220, 38, 38)        # Blood ruby talisman

# Entities
COLOR_HUNTER: tuple[int, int, int] = (153, 218, 199)          # Verdigris under the pilgrim's boots
COLOR_HUNTER_RING: tuple[int, int, int] = (255, 181, 69)      # Light from the brass lantern
COLOR_GHOST: tuple[int, int, int] = (218, 231, 222)           # Moonlit burial linen
COLOR_GHOST_AURA: tuple[int, int, int] = (169, 217, 221)      # Cold light around the shroud
COLOR_TARGET_MARKER: tuple[int, int, int] = (210, 87, 73)     # Enamel on the MAP sight

# A* Path Glow
COLOR_ASTAR_PATH: tuple[int, int, int] = (249, 115, 22)       # Hellfire Trail Orange
COLOR_ASTAR_PATH_PULSE: tuple[int, int, int] = (253, 224, 71) # Molten Flare Yellow

# Sidebar Cards (Horror Aesthetic)
CARD1_BG: tuple[int, int, int] = (18, 12, 24)
CARD1_BORDER: tuple[int, int, int] = (109, 40, 217)
CARD1_TITLE: tuple[int, int, int] = (255, 143, 163)             # Blood Rose Title
TAG_PILL_BG: tuple[int, int, int] = (76, 29, 149)             # Necrotic Violet
TAG_PILL_TEXT: tuple[int, int, int] = (237, 233, 254)
STATUS_ACTIVE_BG: tuple[int, int, int] = (20, 83, 45)         # Acid Green Active
STATUS_ACTIVE_TEXT: tuple[int, int, int] = (187, 247, 208)
BTN_RESTART_BG: tuple[int, int, int] = (88, 28, 135)          # Deep Violet Button
BTN_RESTART_HOVER: tuple[int, int, int] = (126, 34, 206)

# Card 2: Mission Telemetry
CARD2_BG: tuple[int, int, int] = (16, 14, 26)
CARD2_BORDER: tuple[int, int, int] = (76, 29, 149)
CARD2_TITLE: tuple[int, int, int] = (132, 226, 255)
CARD2_LABEL: tuple[int, int, int] = (192, 218, 250)
CARD2_VALUE: tuple[int, int, int] = (151, 235, 255)

# Card 3: HMM Belief State
CARD3_BG: tuple[int, int, int] = (24, 13, 20)
CARD3_BORDER: tuple[int, int, int] = (159, 18, 57)
CARD3_TITLE: tuple[int, int, int] = (251, 113, 133)
CARD3_LABEL: tuple[int, int, int] = (250, 174, 200)
CARD3_VALUE: tuple[int, int, int] = (254, 205, 211)

# Card 4: Controls & Status
CARD4_BG: tuple[int, int, int] = (15, 15, 25)
CARD4_BORDER: tuple[int, int, int] = (76, 29, 149)
CARD4_TITLE: tuple[int, int, int] = (192, 132, 252)
CARD4_KEY: tuple[int, int, int] = (255, 156, 180)
CARD4_DESC: tuple[int, int, int] = (218, 205, 255)

# Buttons in Menus
BTN_PRIMARY_BG: tuple[int, int, int] = (100, 12, 38)          # Crimson Blood Button
BTN_PRIMARY_HOVER: tuple[int, int, int] = (129, 16, 45)       # Bright Blood
BTN_SECONDARY_BG: tuple[int, int, int] = (54, 21, 98)        # Deep Purple Slate
BTN_SECONDARY_HOVER: tuple[int, int, int] = (76, 29, 149)    # Radiant Purple
BTN_DANGER_BG: tuple[int, int, int] = (127, 29, 29)           # Dark Carmine Exit
BTN_DANGER_HOVER: tuple[int, int, int] = (125, 20, 32)

# Legacy thermal placeholders (backward compatibility)
HEATMAP_P0: tuple[int, int, int] = (14, 16, 26)
HEATMAP_P1: tuple[int, int, int] = (22, 163, 74)
HEATMAP_P2: tuple[int, int, int] = (217, 119, 6)
HEATMAP_P3: tuple[int, int, int] = (185, 28, 28)
