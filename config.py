"""
Global Configuration and Constants for Ghost Hunter.
Curated High-Contrast Palette inspired by the 'Sea Escape' & 'Midnight Luxe' 
cool oceanic radar tones paired with a vivid thermal heatmap progression.
"""

# Grid & Dynamics Parameters
GRID_WIDTH: int = 10
GRID_HEIGHT: int = 10
TURN_LIMIT_T_MAX: int = 100

# Sensor & Probability Defaults
DEFAULT_SENSOR_SIGMA: float = 1.0
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
# Curated Palette: "Deep Sea Cyber-Radar"
# -----------------------------------------------------------------------------
# Deepest Oceanic Navy for Canvas
COLOR_BG: tuple[int, int, int] = (10, 17, 30)                # Deep Sapphire / Void
COLOR_GRID_SURFACE: tuple[int, int, int] = (16, 26, 44)       # Clean dark slate cell
COLOR_GRID_LINE: tuple[int, int, int] = (25, 39, 64)          # Crisp boundary line
COLOR_WALL: tuple[int, int, int] = (6, 11, 20)                # Solid dark obstruction
COLOR_WALL_ACCENT: tuple[int, int, int] = (40, 58, 90)        # Distinct wall beveled highlight

# Entities (High visibility)
COLOR_HUNTER: tuple[int, int, int] = (0, 212, 255)            # Radiant Aqua Wave (Hunter)
COLOR_HUNTER_GLOW: tuple[int, int, int] = (120, 240, 255)     # Soft Arctic Breeze aura
COLOR_GHOST: tuple[int, int, int] = (255, 75, 110)            # Vivid Coral Rose (Ghost)
COLOR_GHOST_GLOW: tuple[int, int, int] = (255, 130, 160)      # Ghost ethereal aura
COLOR_TARGET_MARKER: tuple[int, int, int] = (255, 200, 30)    # Solar Gold crosshair

# Sidebar & Dashboard Cards
COLOR_PANEL_BG: tuple[int, int, int] = (10, 17, 30)           # Matching seamless canvas
COLOR_CARD_BG: tuple[int, int, int] = (18, 29, 50)            # Elevated Deep Sea card
COLOR_CARD_BORDER: tuple[int, int, int] = (35, 54, 88)        # Crisp border stroke
COLOR_BTN_BG: tuple[int, int, int] = (14, 116, 144)           # Deep Cyan Button
COLOR_BTN_HOVER: tuple[int, int, int] = (6, 182, 212)         # Bright Aqua on hover

# Text Elements
COLOR_TEXT_PRIMARY: tuple[int, int, int] = (245, 250, 255)    # Crisp Arctic White
COLOR_TEXT_SECONDARY: tuple[int, int, int] = (140, 168, 205)  # Muted Periwinkle
COLOR_TEXT_MUTED: tuple[int, int, int] = (90, 115, 150)       # Dark slate
COLOR_ACCENT_CYAN: tuple[int, int, int] = (34, 211, 238)      # Aqua highlight
COLOR_SUCCESS: tuple[int, int, int] = (52, 211, 153)          # Emerald Mint status

# 4-Stage Thermal Heatmap:
# Low prob (Cool Deep Ocean) -> Mid (Teal Aqua) -> Hot (Golden Sun) -> Peak (Fiery Coral Red)
HEATMAP_P0: tuple[int, int, int] = (16, 26, 44)               # Level 0: Pure empty tile
HEATMAP_P1: tuple[int, int, int] = (14, 80, 120)              # Level 1: Deep Aqua Wave
HEATMAP_P2: tuple[int, int, int] = (220, 140, 20)             # Level 2: Warm Golden Amber
HEATMAP_P3: tuple[int, int, int] = (245, 45, 80)              # Level 3: Peak Thermal Coral Red
