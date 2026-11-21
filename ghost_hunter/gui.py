"""
Pygame Visualizer and Interactive GUI for Ghost Hunter.
Full Horror Redesign:
- Always-on Manhattan Proximity Heatmap centred on HMM MAP estimate (Red=1, Yellow=2/3, Green=4/5)
- Semi-transparent Pause Menu Overlay triggered by [P] (Resume, Change Difficulty, Quit to Menu)
- Authentic Spooky / Horror Art Design: Custom drawn Ghost silhouette, Hunter flashlight, Cursed stone walls, Blood drips & Cobwebs
- Dark atmospheric sidebar telemetry and controls
- Dynamic screen resizing and full-screen auto-scaling without empty margins
"""

from typing import Optional
import math
import time
import sys
import ctypes
import pygame
import numpy as np
from config import (
    GRID_WIDTH,
    GRID_HEIGHT,
    DIFFICULTY_CONFIGS,
    ASTAR_HINT_DURATION_SEC,
    ASTAR_HINT_COOLDOWN_SEC,
    GHOST_MOVE_INTERVAL_SEC,
    COLOR_WINDOW_BG,
    COLOR_GRID_OUTER_FRAME,
    COLOR_GRID_BG,
    COLOR_GRID_SURFACE,
    COLOR_GRID_LINE,
    HEATMAP_DIST_1,
    HEATMAP_DIST_2_3,
    HEATMAP_DIST_4_5,
    COLOR_WALL,
    COLOR_WALL_INNER,
    COLOR_WALL_ACCENT,
    COLOR_WALL_RIVET,
    COLOR_HUNTER,
    COLOR_HUNTER_RING,
    COLOR_GHOST,
    COLOR_GHOST_AURA,
    COLOR_TARGET_MARKER,
    COLOR_ASTAR_PATH,
    COLOR_ASTAR_PATH_PULSE,
    CARD1_BG,
    CARD1_BORDER,
    CARD1_TITLE,
    TAG_PILL_BG,
    TAG_PILL_TEXT,
    STATUS_ACTIVE_BG,
    STATUS_ACTIVE_TEXT,
    BTN_RESTART_BG,
    BTN_RESTART_HOVER,
    CARD2_BG,
    CARD2_BORDER,
    CARD2_TITLE,
    CARD2_LABEL,
    CARD2_VALUE,
    CARD3_BG,
    CARD3_BORDER,
    CARD3_TITLE,
    CARD3_LABEL,
    CARD3_VALUE,
    CARD4_BG,
    CARD4_BORDER,
    CARD4_TITLE,
    CARD4_KEY,
    CARD4_DESC,
    BTN_PRIMARY_BG,
    BTN_PRIMARY_HOVER,
    BTN_SECONDARY_BG,
    BTN_SECONDARY_HOVER,
    BTN_DANGER_BG,
    BTN_DANGER_HOVER,
    MOVES_DICT
)
from ghost_hunter.environment import GridEnvironment
from ghost_hunter.sensor_model import manhattan_distance

# Set Windows process DPI awareness so the OS does not bitmap-stretch the Pygame window
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def lerp_rgb(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Smooth linear interpolation between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t)
    )


class PygameVisualizer:
    """Horror Gothic Aesthetic GUI for Ghost Hunter."""

    def __init__(self, env: GridEnvironment, cell_size: int = 58):
        pygame.init()
        self.env = env
        self.cell_size = cell_size
        self.grid_margin_x = 22
        self.grid_margin_y = 22

        self.grid_pixel_w = env.width * cell_size
        self.grid_pixel_h = env.height * cell_size

        self.sidebar_w = 345
        self.window_w = max(960, self.grid_margin_x * 2 + self.grid_pixel_w + self.sidebar_w + 14)
        self.window_h = max(660, self.grid_margin_y * 2 + self.grid_pixel_h)

        # Fullscreen by default
        self.is_fullscreen = True
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Ghost Hunter · Probabilistic AI Tracking (HMM)")
        self.clock = pygame.time.Clock()

        # Typography: Atmospheric fonts
        title_font = "impact,georgia,arial"
        body_font = "segoeui,arial,sans-serif"
        mono_font = "consolas,courier"

        self.font_hero = pygame.font.SysFont(title_font, 44, bold=True)
        self.font_title = pygame.font.SysFont(title_font, 26, bold=True)
        self.font_subtitle = pygame.font.SysFont(body_font, 14, bold=True)
        self.font_body = pygame.font.SysFont(body_font, 13)
        self.font_body_bold = pygame.font.SysFont(body_font, 13, bold=True)
        self.font_mono = pygame.font.SysFont(mono_font, 13, bold=True)
        self.font_badge = pygame.font.SysFont(body_font, 12, bold=True)
        self.font_btn = pygame.font.SysFont(title_font, 18, bold=True)

        # Game state tracking
        self.reveal_ghost = False
        self.paused = False
        self.start_time = time.time()

        # In-game Card 1 Buttons
        self.restart_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.restart_hovered = False
        self.menu_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.menu_btn_hovered = False

        # Pause Menu Buttons
        self.pause_resume_rect = pygame.Rect(0, 0, 0, 0)
        self.pause_diff_rect = pygame.Rect(0, 0, 0, 0)
        self.pause_quit_rect = pygame.Rect(0, 0, 0, 0)

        # A* Radar ability tracking & Counter
        self.astar_path: list[tuple[int, int]] = []
        self.astar_active_until: float = 0.0
        self.astar_cooldown_until: float = 0.0
        self.astar_use_count: int = 0

        # Periodic ghost glimpse
        self.ghost_glimpse_until: float = 0.0
        self.next_glimpse_time: float = 0.0
        self.glimpse_interval: float = 8.0
        self.glimpse_duration: float = 0.8

        # Continuous smooth movement tracking for key hold (WASD / Arrows)
        self.last_move_time: float = 0.0
        self.move_repeat_delay: float = 0.16

    def resize_for_difficulty(self, env: GridEnvironment, cell_size: int) -> None:
        """Stores new grid environment."""
        self.env = env
        self.cell_size = cell_size
        self.grid_pixel_w = env.width * cell_size
        self.grid_pixel_h = env.height * cell_size
        self.window_w = max(960, self.grid_margin_x * 2 + self.grid_pixel_w + self.sidebar_w + 14)
        self.window_h = max(660, self.grid_margin_y * 2 + self.grid_pixel_h)
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)

    def _compute_layout(self) -> tuple[int, int, int, int, int, int, int]:
        """Dynamically compute layout so the grid and sidebar cleanly span the full screen."""
        scr_w = self.screen.get_width()
        scr_h = self.screen.get_height()
        margin = 18
        gap = 18

        max_grid_h = scr_h - margin * 2
        cs_from_h = max(1, max_grid_h // self.env.height)

        max_grid_w = scr_w - 320 - gap - margin * 2
        cs_from_w = max(1, max_grid_w // self.env.width)

        cs = max(8, min(cs_from_h, cs_from_w))
        gpw = self.env.width * cs
        gph = self.env.height * cs

        ox = margin
        oy = max(margin, (scr_h - gph) // 2)

        sx = ox + gpw + gap
        sb_w = max(280, scr_w - sx - margin)

        return cs, ox, oy, gpw, gph, sx, sb_w

    def toggle_fullscreen(self) -> None:
        """Toggles between fullscreen and windowed mode."""
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)

    def is_astar_active(self) -> bool:
        """Returns True if the A* path hint is currently illuminated."""
        return time.time() < self.astar_active_until

    def trigger_astar_hint(self, path: list[tuple[int, int]]) -> bool:
        """Activates A* path hint for 3s if cooldown (5s) has expired."""
        now = time.time()
        if now < self.astar_cooldown_until:
            return False
        self.astar_use_count += 1
        self.astar_path = path
        self.astar_active_until = now + ASTAR_HINT_DURATION_SEC
        self.astar_cooldown_until = now + ASTAR_HINT_DURATION_SEC + ASTAR_HINT_COOLDOWN_SEC
        return True

    def get_astar_status(self) -> tuple[str, float]:
        """Returns ('ACTIVE', remaining_sec) or ('COOLDOWN', remaining_sec) or ('READY', total_uses)."""
        now = time.time()
        if now < self.astar_active_until:
            return "ACTIVE", max(0.0, self.astar_active_until - now)
        elif now < self.astar_cooldown_until:
            return "COOLDOWN", max(0.0, self.astar_cooldown_until - now)
        else:
            return "READY", float(self.astar_use_count)

    def is_ghost_visible(self) -> bool:
        """Ghost visibility check."""
        return self.env.captured or self.reveal_ghost or (time.time() < self.ghost_glimpse_until)

    def handle_events(self) -> tuple[bool, Optional[tuple[int, int]], bool, bool, bool, Optional[str]]:
        """
        Polls user input in active gameplay.
        Returns:
            (running, human_action, restart_triggered, menu_triggered, astar_requested, pause_action)
            pause_action can be: None, 'resume', 'change_difficulty', 'quit_to_menu'
        """
        human_action = None
        restart_triggered = False
        menu_triggered = False
        astar_requested = False
        pause_action = None

        mouse_pos = pygame.mouse.get_pos()
        self.restart_hovered = self.restart_btn_rect.collidepoint(mouse_pos)
        self.menu_btn_hovered = self.menu_btn_rect.collidepoint(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, None, False, False, False, None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.paused:
                        if self.pause_resume_rect.collidepoint(mouse_pos):
                            self.paused = False
                            pause_action = "resume"
                        elif self.pause_diff_rect.collidepoint(mouse_pos):
                            self.paused = False
                            pause_action = "change_difficulty"
                        elif self.pause_quit_rect.collidepoint(mouse_pos):
                            self.paused = False
                            pause_action = "quit_to_menu"
                    else:
                        if self.restart_hovered:
                            restart_triggered = True
                        elif self.menu_btn_hovered:
                            menu_triggered = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.paused:
                        self.paused = False
                    else:
                        menu_triggered = True
                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif not self.paused:
                    if event.key == pygame.K_r:
                        restart_triggered = True
                    elif event.key == pygame.K_x:
                        astar_requested = True
                    elif event.key == pygame.K_g:
                        self.reveal_ghost = not self.reveal_ghost
                    elif event.key in (pygame.K_UP, pygame.K_w):
                        human_action = MOVES_DICT["UP"]
                        self.last_move_time = time.time() + 0.10
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        human_action = MOVES_DICT["DOWN"]
                        self.last_move_time = time.time() + 0.10
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        human_action = MOVES_DICT["LEFT"]
                        self.last_move_time = time.time() + 0.10
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        human_action = MOVES_DICT["RIGHT"]
                        self.last_move_time = time.time() + 0.10
                    elif event.key == pygame.K_PERIOD:
                        human_action = MOVES_DICT["STAY"]

        # Hold-to-glide movement if not paused
        if human_action is None and not self.paused:
            now = time.time()
            if now - self.last_move_time >= self.move_repeat_delay:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    human_action = MOVES_DICT["RIGHT"]
                    self.last_move_time = now
                elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    human_action = MOVES_DICT["LEFT"]
                    self.last_move_time = now
                elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    human_action = MOVES_DICT["DOWN"]
                    self.last_move_time = now
                elif keys[pygame.K_UP] or keys[pygame.K_w]:
                    human_action = MOVES_DICT["UP"]
                    self.last_move_time = now

        return True, human_action, restart_triggered, menu_triggered, astar_requested, pause_action

    def _draw_card(self, rect: pygame.Rect, bg_color: tuple[int, int, int], border_color: tuple[int, int, int]) -> None:
        """Draws rounded gothic container card with double-accent border."""
        pygame.draw.rect(self.screen, bg_color, rect, border_radius=8)
        pygame.draw.rect(self.screen, border_color, rect, width=2, border_radius=8)

    def _draw_wall_tile(self, rx: int, ry: int, cs: int) -> None:
        """Renders cursed stone crypt obstacle wall tile."""
        cell_rect = pygame.Rect(rx, ry, cs, cs)
        pygame.draw.rect(self.screen, COLOR_WALL, cell_rect)
        inner_m = max(2, cs // 10)
        inner_wall = pygame.Rect(rx + inner_m, ry + inner_m, cs - inner_m * 2, cs - inner_m * 2)
        pygame.draw.rect(self.screen, COLOR_WALL_INNER, inner_wall, border_radius=4)
        pygame.draw.rect(self.screen, COLOR_WALL_ACCENT, inner_wall, width=1, border_radius=4)

        if cs >= 18:
            # Crack lines
            pygame.draw.line(self.screen, (50, 20, 60), (rx + inner_m + 2, ry + inner_m + 2), (rx + cs - inner_m - 2, ry + cs - inner_m - 2), 1)
            pygame.draw.line(self.screen, (50, 20, 60), (rx + cs - inner_m - 2, ry + inner_m + 2), (rx + inner_m + 2, ry + cs - inner_m - 2), 1)
            # Talisman ruby
            rivet_r = max(1, cs // 16)
            pygame.draw.circle(self.screen, COLOR_WALL_RIVET, (rx + cs // 2, ry + cs // 2), rivet_r)

    def _draw_hunter_token(self, cx: int, cy: int, token_r: int, pulse: float) -> None:
        """Renders the Hunter with animated flashlight beam and ether aura."""
        aura_r = int(token_r * (1.3 + 0.2 * pulse))
        pygame.draw.circle(self.screen, COLOR_HUNTER_RING, (cx, cy), aura_r, 2)
        pygame.draw.circle(self.screen, COLOR_HUNTER, (cx, cy), token_r)

        # Flashlight cross / core
        arm = max(2, token_r - 2)
        pygame.draw.line(self.screen, (255, 255, 255), (cx - arm, cy), (cx + arm, cy), 2)
        pygame.draw.line(self.screen, (255, 255, 255), (cx, cy - arm), (cx, cy + arm), 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), max(2, token_r // 3))

    def _draw_ghost_token(self, cx: int, cy: int, token_r: int, pulse: float) -> None:
        """Renders authentic floating Ghost phantom with ruffled tail and glowing spectral eyes."""
        bob = int(math.sin(time.time() * 6.0) * 2.5)
        cy = cy + bob

        # Spectral purple aura
        aura_r = int(token_r * (1.4 + 0.15 * pulse))
        pygame.draw.circle(self.screen, COLOR_GHOST_AURA, (cx, cy), aura_r, 2)

        # Ghost head (circle)
        head_r = int(token_r * 0.85)
        pygame.draw.circle(self.screen, COLOR_GHOST, (cx, cy - token_r // 4), head_r)

        # Ghost body skirt (rect)
        skirt_w = int(head_r * 1.8)
        skirt_h = int(token_r * 0.9)
        skirt_rect = pygame.Rect(cx - skirt_w // 2, cy - token_r // 4, skirt_w, skirt_h)
        pygame.draw.rect(self.screen, COLOR_GHOST, skirt_rect)

        # Scalloped wavy bottom tails
        scallop_r = max(2, skirt_w // 6)
        y_bot = skirt_rect.bottom
        for i in range(3):
            x_scallop = skirt_rect.left + (2 * i + 1) * scallop_r
            pygame.draw.circle(self.screen, COLOR_GHOST, (x_scallop, y_bot), scallop_r)

        # Glowing Spectral Eyes
        eye_offset_x = max(2, head_r // 3)
        eye_y = cy - token_r // 3
        eye_r = max(2, head_r // 4)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx - eye_offset_x, eye_y), eye_r)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx + eye_offset_x, eye_y), eye_r)
        # Deep crimson pupils
        pygame.draw.circle(self.screen, (150, 0, 0), (cx - eye_offset_x, eye_y), max(1, eye_r // 2))
        pygame.draw.circle(self.screen, (150, 0, 0), (cx + eye_offset_x, eye_y), max(1, eye_r // 2))

    def _draw_cobweb(self, ox: int, oy: int, size: int, corner: str = "TL") -> None:
        """Draws creepy cobweb motif in corner."""
        color = (55, 45, 65)
        steps = 4
        for i in range(1, steps + 1):
            r = int(size * (i / steps))
            if corner == "TL":
                pygame.draw.arc(self.screen, color, (ox - r, oy - r, r * 2, r * 2), 0, math.pi / 2, 1)
            elif corner == "TR":
                pygame.draw.arc(self.screen, color, (ox - r, oy - r, r * 2, r * 2), math.pi / 2, math.pi, 1)
        # Web rays
        if corner == "TL":
            pygame.draw.line(self.screen, color, (ox, oy), (ox + size, oy), 1)
            pygame.draw.line(self.screen, color, (ox, oy), (ox, oy + size), 1)
            pygame.draw.line(self.screen, color, (ox, oy), (ox + int(size * 0.7), oy + int(size * 0.7)), 1)
        elif corner == "TR":
            pygame.draw.line(self.screen, color, (ox, oy), (ox - size, oy), 1)
            pygame.draw.line(self.screen, color, (ox, oy), (ox, oy + size), 1)
            pygame.draw.line(self.screen, color, (ox, oy), (ox - int(size * 0.7), oy + int(size * 0.7)), 1)

    def draw_pause_overlay(self) -> None:
        """Renders semi-transparent horror pause menu overlay."""
        scr_w = self.screen.get_width()
        scr_h = self.screen.get_height()

        # Dark translucent overlay
        overlay = pygame.Surface((scr_w, scr_h), pygame.SRCALPHA)
        overlay.fill((5, 5, 10, 195))
        self.screen.blit(overlay, (0, 0))

        # Pause Modal Card
        card_w, card_h = 440, 290
        modal_rect = pygame.Rect((scr_w - card_w) // 2, (scr_h - card_h) // 2, card_w, card_h)
        pygame.draw.rect(self.screen, (15, 10, 22), modal_rect, border_radius=12)
        pygame.draw.rect(self.screen, (159, 18, 57), modal_rect, width=3, border_radius=12)

        # Title
        p_title = self.font_hero.render("PAUSED", True, (244, 63, 94))
        self.screen.blit(p_title, (modal_rect.centerx - p_title.get_width() // 2, modal_rect.y + 18))

        sub_msg = self.font_subtitle.render("The phantoms wait in the shadows...", True, (168, 85, 247))
        self.screen.blit(sub_msg, (modal_rect.centerx - sub_msg.get_width() // 2, modal_rect.y + 68))

        mouse_pos = pygame.mouse.get_pos()
        btn_w, btn_h = 320, 44
        btn_x = modal_rect.centerx - btn_w // 2

        self.pause_resume_rect = pygame.Rect(btn_x, modal_rect.y + 104, btn_w, btn_h)
        self.pause_diff_rect   = pygame.Rect(btn_x, modal_rect.y + 158, btn_w, btn_h)
        self.pause_quit_rect   = pygame.Rect(btn_x, modal_rect.y + 212, btn_w, btn_h)

        h_resume = self.pause_resume_rect.collidepoint(mouse_pos)
        h_diff   = self.pause_diff_rect.collidepoint(mouse_pos)
        h_quit   = self.pause_quit_rect.collidepoint(mouse_pos)

        # Button 1: Resume
        c_res = BTN_PRIMARY_HOVER if h_resume else BTN_PRIMARY_BG
        pygame.draw.rect(self.screen, c_res, self.pause_resume_rect, border_radius=8)
        t_res = self.font_btn.render("RESUME [P]", True, (255, 255, 255))
        self.screen.blit(t_res, (self.pause_resume_rect.centerx - t_res.get_width() // 2, self.pause_resume_rect.centery - t_res.get_height() // 2))

        # Button 2: Change Difficulty
        c_diff = BTN_SECONDARY_HOVER if h_diff else BTN_SECONDARY_BG
        pygame.draw.rect(self.screen, c_diff, self.pause_diff_rect, border_radius=8)
        t_diff = self.font_btn.render("CHANGE DIFFICULTY", True, (255, 255, 255))
        self.screen.blit(t_diff, (self.pause_diff_rect.centerx - t_diff.get_width() // 2, self.pause_diff_rect.centery - t_diff.get_height() // 2))

        # Button 3: Quit to Menu
        c_quit = BTN_DANGER_HOVER if h_quit else BTN_DANGER_BG
        pygame.draw.rect(self.screen, c_quit, self.pause_quit_rect, border_radius=8)
        t_quit = self.font_btn.render("QUIT TO MAIN MENU", True, (255, 255, 255))
        self.screen.blit(t_quit, (self.pause_quit_rect.centerx - t_quit.get_width() // 2, self.pause_quit_rect.centery - t_quit.get_height() // 2))

    def render(self, belief_grid: Optional[np.ndarray] = None, mode_label: str = "Demo") -> None:
        """Renders grid canvas, horror tokens, A* path overlay, and telemetry."""
        self.screen.fill(COLOR_WINDOW_BG)

        elapsed = time.time() - self.start_time
        pulse = 0.5 + 0.5 * math.sin(elapsed * 5.0)

        cs, ox, oy, gpw, gph, sx, card_w = self._compute_layout()

        # Find MAP estimate (argmax of belief grid)
        map_cell: Optional[tuple[int, int]] = None
        if belief_grid is not None:
            raw_map = np.unravel_index(np.argmax(belief_grid), belief_grid.shape)
            map_cell = (int(raw_map[0]), int(raw_map[1]))

        is_astar_active = time.time() < self.astar_active_until
        astar_cells = set(self.astar_path) if is_astar_active else set()

        # -------------------------------------------------------------
        # 1. Grid Canvas Area
        # -------------------------------------------------------------
        grid_frame = pygame.Rect(ox - 3, oy - 3, gpw + 6, gph + 6)
        pygame.draw.rect(self.screen, COLOR_GRID_BG, grid_frame, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_GRID_OUTER_FRAME, grid_frame, width=2, border_radius=10)

        for r in range(self.env.height):
            for c in range(self.env.width):
                rx = ox + c * cs
                ry = oy + r * cs
                cell_rect = pygame.Rect(rx, ry, cs, cs)

                if (r, c) in self.env.walls:
                    self._draw_wall_tile(rx, ry, cs)
                else:
                    # Manhattan Distance-based Proximity Heatmap from HMM MAP estimate
                    # Distance 1: Red | Distance 2, 3: Yellow/Amber | Distance 4, 5: Green
                    tile_color = COLOR_GRID_SURFACE
                    if map_cell is not None:
                        dist = abs(r - map_cell[0]) + abs(c - map_cell[1])
                        if dist == 1:
                            tile_color = HEATMAP_DIST_1
                        elif dist in (2, 3):
                            tile_color = HEATMAP_DIST_2_3
                        elif dist in (4, 5):
                            tile_color = HEATMAP_DIST_4_5

                    # A* path glow illumination
                    if is_astar_active and (r, c) in astar_cells:
                        path_glow = lerp_rgb(COLOR_ASTAR_PATH, COLOR_ASTAR_PATH_PULSE, pulse)
                        tile_color = lerp_rgb(tile_color, path_glow, 0.85)

                    pygame.draw.rect(self.screen, tile_color, cell_rect)
                    pygame.draw.rect(self.screen, COLOR_GRID_LINE, cell_rect, width=1)

                    if is_astar_active and (r, c) in astar_cells:
                        dot_x = rx + cs // 2
                        dot_y = ry + cs // 2
                        pygame.draw.circle(self.screen, (255, 255, 255), (dot_x, dot_y), max(2, cs // 8))

        # -------------------------------------------------------------
        # 2. Hunter Token
        # -------------------------------------------------------------
        hr, hc = self.env.hunter_pos
        hx = ox + hc * cs + cs // 2
        hy = oy + hr * cs + cs // 2
        token_r = max(4, cs // 3)
        self._draw_hunter_token(hx, hy, token_r, pulse)

        # -------------------------------------------------------------
        # 3. Ghost Token (visible on capture, [G] reveal, or 8s periodic glimpse)
        # -------------------------------------------------------------
        if self.is_ghost_visible():
            gr, gc = self.env.true_ghost_pos
            gx = ox + gc * cs + cs // 2
            gy = oy + gr * cs + cs // 2
            ghost_token_r = max(4, cs // 3)
            self._draw_ghost_token(gx, gy, ghost_token_r, pulse)

        # -------------------------------------------------------------
        # 3b. Target Reticle — always marks HMM MAP estimate
        # -------------------------------------------------------------
        if map_cell is not None:
            tr, tc = map_cell
            tx = ox + tc * cs + cs // 2
            ty = oy + tr * cs + cs // 2
            reticle_r = max(6, int(cs * 0.38))
            pygame.draw.circle(self.screen, COLOR_TARGET_MARKER, (tx, ty), reticle_r, 2)
            gap = max(2, reticle_r // 3)
            arm = max(4, reticle_r + 4)
            pygame.draw.line(self.screen, COLOR_TARGET_MARKER, (tx, ty - gap), (tx, ty - arm), 2)
            pygame.draw.line(self.screen, COLOR_TARGET_MARKER, (tx, ty + gap), (tx, ty + arm), 2)
            pygame.draw.line(self.screen, COLOR_TARGET_MARKER, (tx - gap, ty), (tx - arm, ty), 2)
            pygame.draw.line(self.screen, COLOR_TARGET_MARKER, (tx + gap, ty), (tx + arm, ty), 2)

        # -------------------------------------------------------------
        # 4. Sidebar Cards
        # -------------------------------------------------------------
        sy = oy
        total_sidebar_h = gph
        gap = 10
        total_gaps = gap * 3
        usable_h = total_sidebar_h - total_gaps

        c1_h = max(110, int(usable_h * 0.20))
        c2_h = max(120, int(usable_h * 0.22))
        c3_h = max(135, int(usable_h * 0.24))
        c4_h = usable_h - c1_h - c2_h - c3_h

        # --- Card 1: GHOST HUNTER Header Card ---
        card1_rect = pygame.Rect(sx, sy, card_w, c1_h)
        self._draw_card(card1_rect, CARD1_BG, CARD1_BORDER)

        title_surf = self.font_title.render("GHOST HUNTER", True, CARD1_TITLE)
        self.screen.blit(title_surf, (sx + 14, sy + 8))

        # Mode Pill
        tag_text = f"{self.env.width}x{self.env.height} · {mode_label.upper()}"
        tag_bg = pygame.Rect(sx + 14, sy + 38, 160, 24)
        pygame.draw.rect(self.screen, TAG_PILL_BG, tag_bg, border_radius=6)
        tag_surf = self.font_badge.render(tag_text, True, TAG_PILL_TEXT)
        self.screen.blit(tag_surf, (tag_bg.centerx - tag_surf.get_width() // 2, tag_bg.centery - tag_surf.get_height() // 2))

        # Status Pill
        status_text = "CAPTURED!" if self.env.captured else ("PAUSED" if self.paused else "HUNTING")
        st_bg_color = (159, 18, 57) if self.env.captured else ((180, 83, 9) if self.paused else STATUS_ACTIVE_BG)
        st_txt_color = (255, 228, 230) if self.env.captured else ((254, 240, 138) if self.paused else STATUS_ACTIVE_TEXT)
        st_bg = pygame.Rect(sx + card_w - 95, sy + 38, 82, 24)
        pygame.draw.rect(self.screen, st_bg_color, st_bg, border_radius=6)
        st_surf = self.font_badge.render(status_text, True, st_txt_color)
        self.screen.blit(st_surf, (st_bg.centerx - st_surf.get_width() // 2, st_bg.centery - st_surf.get_height() // 2))

        # Restart & Menu Buttons (clean horror buttons)
        btn_y = sy + 70
        btn_h = max(24, c1_h - 78)
        self.restart_btn_rect = pygame.Rect(sx + 14, btn_y, (card_w - 34) // 2 + 10, btn_h)
        btn_bg = BTN_PRIMARY_HOVER if self.restart_hovered else BTN_PRIMARY_BG
        pygame.draw.rect(self.screen, btn_bg, self.restart_btn_rect, border_radius=6)
        btn_surf = self.font_badge.render("RESTART [R]", True, (255, 255, 255))
        self.screen.blit(btn_surf, (self.restart_btn_rect.centerx - btn_surf.get_width() // 2, self.restart_btn_rect.centery - btn_surf.get_height() // 2))

        self.menu_btn_rect = pygame.Rect(self.restart_btn_rect.right + 8, btn_y, card_w - self.restart_btn_rect.width - 36, btn_h)
        m_bg = BTN_SECONDARY_HOVER if self.menu_btn_hovered else BTN_SECONDARY_BG
        pygame.draw.rect(self.screen, m_bg, self.menu_btn_rect, border_radius=6)
        menu_surf = self.font_badge.render("MENU [Esc]", True, (255, 255, 255))
        self.screen.blit(menu_surf, (self.menu_btn_rect.centerx - menu_surf.get_width() // 2, self.menu_btn_rect.centery - menu_surf.get_height() // 2))

        # --- Card 2: MISSION TELEMETRY ---
        sy += c1_h + gap
        card2_rect = pygame.Rect(sx, sy, card_w, c2_h)
        self._draw_card(card2_rect, CARD2_BG, CARD2_BORDER)
        t_lbl = self.font_subtitle.render("MISSION TELEMETRY", True, CARD2_TITLE)
        self.screen.blit(t_lbl, (sx + 14, sy + 8))

        telemetry_rows = [
            ("Turn Step",       f"{self.env.turn} / {self.env.turn_limit}"),
            ("Sensor Reading",  f"{self.env.last_reading} units"),
            ("Sensor Noise (σ)", f"{self.env.sensor_sigma:.1f}"),
            ("Hunter Position", f"({int(self.env.hunter_pos[0])}, {int(self.env.hunter_pos[1])})"),
            ("A* Uses Count",   f"{self.astar_use_count} times")
        ]
        row_step = max(18, (c2_h - 36) // len(telemetry_rows))
        row_y = sy + 32
        for lbl, val in telemetry_rows:
            l_surf = self.font_body.render(lbl, True, CARD2_LABEL)
            v_surf = self.font_mono.render(val, True, CARD2_VALUE)
            self.screen.blit(l_surf, (sx + 14, row_y))
            self.screen.blit(v_surf, (sx + card_w - v_surf.get_width() - 14, row_y))
            row_y += row_step

        # --- Card 3: HMM BELIEF STATE ---
        sy += c2_h + gap
        card3_rect = pygame.Rect(sx, sy, card_w, c3_h)
        self._draw_card(card3_rect, CARD3_BG, CARD3_BORDER)
        h_lbl = self.font_subtitle.render("HMM BELIEF STATE", True, CARD3_TITLE)
        self.screen.blit(h_lbl, (sx + 14, sy + 8))

        if belief_grid is not None and map_cell is not None:
            max_prob = float(np.max(belief_grid))
            map_pos = f"({map_cell[0]}, {map_cell[1]})"
            true_dist = manhattan_distance(self.env.hunter_pos, self.env.true_ghost_pos)
            p = belief_grid[belief_grid > 0]
            entropy = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0
            hmm_rows = [
                ("Most Likely Cell",  map_pos),
                ("Peak Confidence",   f"{max_prob * 100:.1f}%"),
                ("Belief Entropy",    f"{entropy:.2f} bits"),
                ("True Distance",     f"{true_dist if (self.reveal_ghost or self.env.captured) else 'Hidden (G)'}"),
                ("Ghost Position",    f"{self.env.true_ghost_pos if (self.reveal_ghost or self.env.captured) else 'Hidden (G)'}")
            ]
            row_step3 = max(18, (c3_h - 36) // len(hmm_rows))
            h_y = sy + 30
            for lbl, val in hmm_rows:
                l_surf = self.font_body.render(lbl, True, CARD3_LABEL)
                v_surf = self.font_mono.render(val, True, CARD3_VALUE)
                self.screen.blit(l_surf, (sx + 14, h_y))
                self.screen.blit(v_surf, (sx + card_w - v_surf.get_width() - 14, h_y))
                h_y += row_step3

        # --- Card 4: KEYBOARD SHORTCUTS & A* RADAR ---
        sy += c3_h + gap
        card4_rect = pygame.Rect(sx, sy, card_w, c4_h)
        self._draw_card(card4_rect, CARD4_BG, CARD4_BORDER)

        radar_status, status_val = self.get_astar_status()
        if radar_status == "ACTIVE":
            radar_badge = f"ACTIVE ({status_val:.1f}s) | USES: {self.astar_use_count}"
            radar_badge_color = (217, 119, 6)
        elif radar_status == "COOLDOWN":
            radar_badge = f"RECHARGING ({status_val:.1f}s) | USES: {self.astar_use_count}"
            radar_badge_color = (159, 18, 57)
        else:
            radar_badge = f"READY [X] | USES: {self.astar_use_count}"
            radar_badge_color = (16, 185, 129)

        k_lbl = self.font_subtitle.render("CONTROLS & SHORTCUTS", True, CARD4_TITLE)
        self.screen.blit(k_lbl, (sx + 14, sy + 8))

        radar_surf   = self.font_badge.render(radar_badge, True, (255, 255, 255))
        r_pill_rect  = pygame.Rect(sx + card_w - radar_surf.get_width() - 20, sy + 6, radar_surf.get_width() + 12, 20)
        pygame.draw.rect(self.screen, radar_badge_color, r_pill_rect, border_radius=6)
        self.screen.blit(radar_surf, (r_pill_rect.centerx - radar_surf.get_width() // 2, r_pill_rect.centery - radar_surf.get_height() // 2))

        shortcuts = [
            ("[X]",        f"A* Hellfire Radar (Used: {self.astar_use_count})"),
            ("[P]",        "Pause Menu (Overlay)"),
            ("[G]",        "Reveal Ghost Debug"),
            ("[R]",        "Restart & New Crypt"),
            ("[F]",        "Toggle Fullscreen"),
            ("[W,A,S,D]",  "Glide Movement (Hold Key)")
        ]
        sc_step = max(17, (c4_h - 36) // len(shortcuts))
        sc_y = sy + 32
        for k, desc in shortcuts:
            k_surf = self.font_mono.render(k, True, CARD4_KEY)
            d_surf = self.font_body.render(desc, True, CARD4_DESC)
            self.screen.blit(k_surf, (sx + 14, sc_y))
            self.screen.blit(d_surf, (sx + 88, sc_y))
            sc_y += sc_step

        # If game is paused, draw the dark horror overlay on top
        if self.paused:
            self.draw_pause_overlay()

        pygame.display.flip()

    # -------------------------------------------------------------------------
    # Screen 1: Horror Start Menu
    # -------------------------------------------------------------------------
    def show_start_screen(self) -> str:
        """Renders Spooky Horror Start Screen with blood drips and glowing title."""
        btn_w, btn_h = 320, 52

        while True:
            scr_w = self.screen.get_width()
            scr_h = self.screen.get_height()
            center_x = scr_w // 2

            btn_start = pygame.Rect(center_x - btn_w // 2, int(scr_h * 0.44), btn_w, btn_h)
            btn_menu  = pygame.Rect(center_x - btn_w // 2, int(scr_h * 0.44) + 68, btn_w, btn_h)
            btn_exit  = pygame.Rect(center_x - btn_w // 2, int(scr_h * 0.44) + 136, btn_w, btn_h)

            mouse_pos = pygame.mouse.get_pos()
            hover_start = btn_start.collidepoint(mouse_pos)
            hover_menu  = btn_menu.collidepoint(mouse_pos)
            hover_exit  = btn_exit.collidepoint(mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "exit"
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if hover_start:
                        return "start"
                    elif hover_menu:
                        return "instructions"
                    elif hover_exit:
                        return "exit"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        self.toggle_fullscreen()
                    elif event.key == pygame.K_ESCAPE:
                        return "exit"
                    elif event.key == pygame.K_RETURN:
                        return "start"

            self.screen.fill(COLOR_WINDOW_BG)

            # Atmospheric Cobwebs in top corners
            self._draw_cobweb(0, 0, 90, "TL")
            self._draw_cobweb(scr_w, 0, 90, "TR")

            # Blood Drips animated along top edge
            t_now = time.time()
            drip_xs = [center_x - 280, center_x - 140, center_x + 140, center_x + 280]
            for idx, dx in enumerate(drip_xs):
                drip_len = int(24 + 14 * math.sin(t_now * 3.0 + idx * 1.5))
                pygame.draw.line(self.screen, (159, 18, 57), (dx, 0), (dx, drip_len), 3)
                pygame.draw.circle(self.screen, (225, 29, 72), (dx, drip_len), 4)

            # Horror Title with bobbing & blood glow
            bob_y = int(math.sin(t_now * 2.5) * 3)
            hero_title = self.font_hero.render("GHOST HUNTER", True, (244, 63, 94))
            sub_hero = self.font_subtitle.render("Probabilistic State Estimation & Tracking with Hidden Markov Models", True, (168, 85, 247))

            self.screen.blit(hero_title, (center_x - hero_title.get_width() // 2, int(scr_h * 0.18) + bob_y))
            self.screen.blit(sub_hero, (center_x - sub_hero.get_width() // 2, int(scr_h * 0.28)))

            # Button 1: START (Blood Red)
            c_start = BTN_PRIMARY_HOVER if hover_start else BTN_PRIMARY_BG
            pygame.draw.rect(self.screen, c_start, btn_start, border_radius=8)
            pygame.draw.rect(self.screen, (244, 63, 94), btn_start, width=2, border_radius=8)
            txt_s = self.font_btn.render("ENTER THE CRYPT (START)", True, (255, 255, 255))
            self.screen.blit(txt_s, (btn_start.centerx - txt_s.get_width() // 2, btn_start.centery - txt_s.get_height() // 2))

            # Button 2: MAIN MENU (Cursed Violet)
            c_menu = BTN_SECONDARY_HOVER if hover_menu else BTN_SECONDARY_BG
            pygame.draw.rect(self.screen, c_menu, btn_menu, border_radius=8)
            pygame.draw.rect(self.screen, (139, 92, 246), btn_menu, width=2, border_radius=8)
            txt_m = self.font_btn.render("GRIMOIRE (INSTRUCTIONS)", True, (255, 255, 255))
            self.screen.blit(txt_m, (btn_menu.centerx - txt_m.get_width() // 2, btn_menu.centery - txt_m.get_height() // 2))

            # Button 3: EXIT (Obsidian Carmine)
            c_exit = BTN_DANGER_HOVER if hover_exit else BTN_DANGER_BG
            pygame.draw.rect(self.screen, c_exit, btn_exit, border_radius=8)
            pygame.draw.rect(self.screen, (185, 28, 28), btn_exit, width=2, border_radius=8)
            txt_e = self.font_btn.render("FLEE (EXIT)", True, (255, 255, 255))
            self.screen.blit(txt_e, (btn_exit.centerx - txt_e.get_width() // 2, btn_exit.centery - txt_e.get_height() // 2))

            # Footer
            hint_fs = self.font_body.render("Press [F] to toggle Fullscreen  |  [P] Pause in-game", True, (100, 100, 130))
            footer = self.font_body.render("Track the unseen phantom before your sanity expires...", True, (140, 100, 160))
            self.screen.blit(footer, (center_x - footer.get_width() // 2, scr_h - 55))
            self.screen.blit(hint_fs, (center_x - hint_fs.get_width() // 2, scr_h - 32))

            pygame.display.flip()
            self.clock.tick(30)

    # -------------------------------------------------------------------------
    # Screen 2: Grimoire & Instructions Screen
    # -------------------------------------------------------------------------
    def show_instructions_screen(self) -> None:
        """Renders Grimoire instructions modal in gothic theme."""
        while True:
            scr_w = self.screen.get_width()
            scr_h = self.screen.get_height()
            center_x = scr_w // 2
            btn_back = pygame.Rect(center_x - 100, scr_h - 65, 200, 42)

            mouse_pos = pygame.mouse.get_pos()
            hover_back = btn_back.collidepoint(mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and hover_back:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        self.toggle_fullscreen()
                    elif event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                        return

            self.screen.fill(COLOR_WINDOW_BG)
            self._draw_cobweb(0, 0, 70, "TL")
            self._draw_cobweb(scr_w, 0, 70, "TR")

            t = self.font_title.render("GRIMOIRE - SURVIVAL RULES & INSTRUCTIONS", True, (244, 63, 94))
            self.screen.blit(t, (center_x - t.get_width() // 2, 22))

            # Card 1: What Happens
            card1 = pygame.Rect(40, 65, scr_w - 80, 130)
            self._draw_card(card1, CARD1_BG, CARD1_BORDER)
            s_title = self.font_subtitle.render("WHAT HAUNTS THIS LABYRINTH?", True, (244, 63, 94))
            self.screen.blit(s_title, (55, 75))

            lines1 = [
                "• An invisible phantom roams dynamically through cursed labyrinth corridors.",
                "• Your sensor receives noisy distance pings clouded by Gaussian sensor uncertainty sigma.",
                "• The Hidden Markov Model (HMM) tracks the most probable phantom cell using Bayesian Forward updates.",
                "• Always-On Proximity Rings indicate distance to MAP: Red (Dist 1), Amber (Dist 2-3), Green (Dist 4-5)."
            ]
            y = 100
            for l in lines1:
                txt = self.font_body.render(l, True, (226, 232, 240))
                self.screen.blit(txt, (55, y))
                y += 20

            # Card 2: How to Play
            card2 = pygame.Rect(40, 205, scr_w - 80, 155)
            self._draw_card(card2, CARD2_BG, CARD2_BORDER)
            p_title = self.font_subtitle.render("HOW TO HUNT & SURVIVE", True, (168, 85, 247))
            self.screen.blit(p_title, (55, 215))

            lines2 = [
                "• Controls: Use W, A, S, D or Arrow Keys to glide continuously across open corridors.",
                "• Capture: Step directly onto the phantom's cell before the turn limit runs out.",
                "• A* Hellfire Radar [X Key]: Illuminates the fastest route directly to the ghost (5s cooldown).",
                "• Pause Overlay [P Key]: Freezes the nightmare and opens the menu without screen locks.",
                "• Glimpse: Every 8 seconds, the ghost flashes visible for a fraction of a second."
            ]
            y = 240
            for l in lines2:
                txt = self.font_body.render(l, True, (226, 232, 240))
                self.screen.blit(txt, (55, y))
                y += 20

            # Card 3: Difficulties
            card3 = pygame.Rect(40, 370, scr_w - 80, 125)
            self._draw_card(card3, CARD3_BG, CARD3_BORDER)
            d_title = self.font_subtitle.render("DIFFICULTY REALMS", True, (251, 113, 133))
            self.screen.blit(d_title, (55, 380))

            lines3 = [
                "• Beginner: 10x10 Crypt (100 Turns) — Phantom moves every 1.0s.",
                "• Intermediate: 15x15 Tomb (150 Turns) — Phantom moves every 0.75s.",
                "• Hard: 20x20 Catacomb (200 Turns) — Phantom moves every 0.55s.",
                "• Extreme: 30x30 Underworld (300 Turns) — Phantom darts at 0.40s!"
            ]
            y = 405
            for l in lines3:
                txt = self.font_body.render(l, True, (254, 205, 211))
                self.screen.blit(txt, (55, y))
                y += 20

            # Back button
            c_back = BTN_SECONDARY_HOVER if hover_back else BTN_SECONDARY_BG
            pygame.draw.rect(self.screen, c_back, btn_back, border_radius=8)
            pygame.draw.rect(self.screen, (139, 92, 246), btn_back, width=2, border_radius=8)
            b_txt = self.font_btn.render("RETURN", True, (255, 255, 255))
            self.screen.blit(b_txt, (btn_back.centerx - b_txt.get_width() // 2, btn_back.centery - b_txt.get_height() // 2))

            pygame.display.flip()
            self.clock.tick(30)

    # -------------------------------------------------------------------------
    # Screen 3: Choose Your Fate (Difficulty Selection)
    # -------------------------------------------------------------------------
    def show_difficulty_selection(self) -> Optional[str]:
        """Renders Difficulty Selection screen in gothic horror aesthetic."""
        card_w = 520
        card_h = 66
        diff_keys = ["Beginner", "Intermediate", "Hard", "Extreme"]

        while True:
            scr_w = self.screen.get_width()
            scr_h = self.screen.get_height()
            center_x = scr_w // 2
            start_y = int(scr_h * 0.26)

            btn_rects = []
            for i, key in enumerate(diff_keys):
                rect = pygame.Rect(center_x - card_w // 2, start_y + i * 84, card_w, card_h)
                btn_rects.append((key, rect))

            btn_back = pygame.Rect(center_x - 100, scr_h - 65, 200, 42)
            mouse_pos = pygame.mouse.get_pos()
            hover_back = btn_back.collidepoint(mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if hover_back:
                        return None
                    for key, rect in btn_rects:
                        if rect.collidepoint(mouse_pos):
                            return key
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        self.toggle_fullscreen()
                    elif event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_1:
                        return "Beginner"
                    elif event.key == pygame.K_2:
                        return "Intermediate"
                    elif event.key == pygame.K_3:
                        return "Hard"
                    elif event.key == pygame.K_4:
                        return "Extreme"

            self.screen.fill(COLOR_WINDOW_BG)
            self._draw_cobweb(0, 0, 80, "TL")
            self._draw_cobweb(scr_w, 0, 80, "TR")

            t = self.font_hero.render("CHOOSE YOUR FATE", True, (244, 63, 94))
            sub = self.font_subtitle.render("Select the realm dimension to configure the haunted labyrinth", True, (168, 85, 247))
            self.screen.blit(t, (center_x - t.get_width() // 2, int(scr_h * 0.10)))
            self.screen.blit(sub, (center_x - sub.get_width() // 2, int(scr_h * 0.19)))

            theme_colors = {
                "Beginner":     ((22, 163, 74),  (34, 197, 94)),
                "Intermediate": ((217, 119, 6),  (245, 158, 11)),
                "Hard":         ((234, 88, 12),  (249, 115, 22)),
                "Extreme":      ((159, 18, 57),  (244, 63, 94))
            }

            for key, rect in btn_rects:
                is_hovered = rect.collidepoint(mouse_pos)
                border_c, glow_c = theme_colors[key]
                bg_color = (24, 14, 30) if is_hovered else (14, 10, 20)

                pygame.draw.rect(self.screen, bg_color, rect, border_radius=8)
                pygame.draw.rect(self.screen, glow_c if is_hovered else border_c, rect, width=2, border_radius=8)

                cfg = DIFFICULTY_CONFIGS[key]
                k_surf = self.font_btn.render(f"{key.upper()} ({cfg['width']}x{cfg['height']} Realm)", True, glow_c)
                self.screen.blit(k_surf, (rect.x + 18, rect.y + 12))

                d_surf = self.font_body.render(cfg['description'], True, (203, 213, 225))
                self.screen.blit(d_surf, (rect.x + 18, rect.y + 38))

                arrow_surf = self.font_btn.render("ENTER ->", True, glow_c if is_hovered else (148, 163, 184))
                self.screen.blit(arrow_surf, (rect.right - arrow_surf.get_width() - 18, rect.centery - arrow_surf.get_height() // 2))

            c_back = BTN_SECONDARY_HOVER if hover_back else BTN_SECONDARY_BG
            pygame.draw.rect(self.screen, c_back, btn_back, border_radius=8)
            pygame.draw.rect(self.screen, (139, 92, 246), btn_back, width=2, border_radius=8)
            b_txt = self.font_btn.render("RETURN", True, (255, 255, 255))
            self.screen.blit(b_txt, (btn_back.centerx - b_txt.get_width() // 2, btn_back.centery - b_txt.get_height() // 2))

            hint = self.font_body.render("[F] Fullscreen  |  Keys [1] Beginner  [2] Intermediate  [3] Hard  [4] Extreme", True, (120, 120, 150))
            self.screen.blit(hint, (center_x - hint.get_width() // 2, scr_h - 30))

            pygame.display.flip()
            self.clock.tick(30)

    def close(self) -> None:
        """Closes Pygame window."""
        pygame.quit()
