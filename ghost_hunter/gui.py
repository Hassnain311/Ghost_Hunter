"""
Pygame Visualizer and Interactive GUI for Ghost Hunter.
Full Menu System & Advanced Features:
- Windows High-DPI Awareness: Enabled via ctypes for crystal-clear, sharp text rendering
- Typography: Crisp serif Georgia / Times New Roman with high-contrast text rendering
- Palette: Soft Silver, Slate Grey, Charcoal & Warm Gold / Amber
- Default State: Both Heatmap and Ghost location are OFF by default; toggled via [H] and [G]
- Yellow Target Crosshair / Reticle: Restored! Displayed over the MAP (most likely cell) or true ghost position when either [H] or [G] is active.
- Start Menu Screen: "Ghost Hunter" Title (clean, no square box), "Start", "Main Menu", "Exit" (No university tags)
- Difficulty Selection: Beginner (10x10), Intermediate (15x15), Hard (20x20), Extreme (30x30)
- Clean ASCII Glyphs: Standard ASCII arrows (->) to prevent missing glyph rectangle boxes (□)
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
    HEATMAP_P0,
    HEATMAP_P1,
    HEATMAP_P2,
    HEATMAP_P3,
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


def compute_gradient_color(prob: float, max_prob: float) -> tuple[int, int, int]:
    """Renders 4-stage thermal heatmap."""
    if max_prob <= 1e-9:
        return HEATMAP_P0

    val = max(0.0, min(1.0, prob / max_prob))

    if val < 0.22:
        return HEATMAP_P0
    elif val < 0.45:
        t = (val - 0.22) / (0.45 - 0.22)
        return lerp_rgb(HEATMAP_P0, HEATMAP_P1, t)
    elif val < 0.75:
        t = (val - 0.45) / (0.75 - 0.45)
        return lerp_rgb(HEATMAP_P1, HEATMAP_P2, t)
    else:
        t = (val - 0.75) / (1.0 - 0.75)
        return lerp_rgb(HEATMAP_P2, HEATMAP_P3, t)


class PygameVisualizer:
    """Refined Times New Roman & Silver-Gold GUI for Ghost Hunter."""

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

        # Start in fullscreen by default; [F] toggles back to windowed
        self.is_fullscreen = True
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Ghost Hunter · Probabilistic AI Tracking (HMM)")
        self.clock = pygame.time.Clock()

        # Typography: Crisp fonts with clean rendering
        font_family = "georgia,timesnewroman,serif"
        self.font_hero = pygame.font.SysFont(font_family, 38, bold=True)
        self.font_title = pygame.font.SysFont(font_family, 24, bold=True)
        self.font_subtitle = pygame.font.SysFont(font_family, 15, bold=True)
        self.font_body = pygame.font.SysFont(font_family, 14)
        self.font_body_bold = pygame.font.SysFont(font_family, 14, bold=True)
        self.font_mono = pygame.font.SysFont("consolas,courier", 13, bold=True)
        self.font_badge = pygame.font.SysFont(font_family, 12, bold=True)
        self.font_btn = pygame.font.SysFont(font_family, 16, bold=True)

        # Default state at game start:
        # - Yellow pointer is visible from the start (tracks HMM MAP estimate)
        # - Heatmap is OFF by default (toggled with [H])
        # - Ghost is invisible by default (revealed on capture or with [G])
        self.show_heatmap = False
        self.reveal_ghost = False
        self.paused = False
        self.start_time = time.time()

        # Button rects
        self.restart_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.restart_hovered = False
        self.menu_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.menu_btn_hovered = False
        self.heatmap_toggle_rect = pygame.Rect(0, 0, 0, 0)
        self.heatmap_toggle_hovered = False

        # A* Radar ability tracking
        self.astar_path: list[tuple[int, int]] = []
        self.astar_active_until: float = 0.0
        self.astar_cooldown_until: float = 0.0

        # Continuous smooth movement tracking for key hold (WASD / Arrows)
        self.last_move_time: float = 0.0
        self.move_repeat_delay: float = 0.16   # Initial delay / repeat rate in seconds (approx 6 steps/sec)

    def resize_for_difficulty(self, env: GridEnvironment, cell_size: int) -> None:
        """Stores the new grid environment. Cell size is recomputed dynamically
        in render() based on actual screen resolution so the grid always fills
        the full screen properly."""
        self.env = env
        self.base_cell_size = cell_size   # fallback / windowed reference size
        self.grid_pixel_w = env.width * cell_size
        self.grid_pixel_h = env.height * cell_size

        self.window_w = max(960, self.grid_margin_x * 2 + self.grid_pixel_w + self.sidebar_w + 14)
        self.window_h = max(660, self.grid_margin_y * 2 + self.grid_pixel_h)
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)

    def _compute_layout(self) -> tuple[int, int, int, int, int, int, int]:
        """Dynamically compute layout to fill the entire screen seamlessly.

        - Grid is vertically centered with top/bottom margin.
        - Cell size is calculated to make the square grid as large as possible.
        - Left margin and gap between grid and sidebar are balanced.
        - Sidebar dynamically spans all remaining width to the right edge with a clean margin,
          so no empty whitespace is left on the right!
        Returns (cell_size, grid_ox, grid_oy, grid_pixel_w, grid_pixel_h, sidebar_sx, sidebar_w).
        """
        scr_w = self.screen.get_width()
        scr_h = self.screen.get_height()
        margin = 18
        gap = 18

        # 1. Height-driven square grid calculation (grid fills full vertical height minus margins)
        max_grid_h = scr_h - margin * 2
        cs_from_h = max(1, max_grid_h // self.env.height)

        # Ensure grid leaves at least 320px for sidebar on the right
        max_grid_w = scr_w - 320 - gap - margin * 2
        cs_from_w = max(1, max_grid_w // self.env.width)

        cs = max(8, min(cs_from_h, cs_from_w))
        gpw = self.env.width * cs
        gph = self.env.height * cs

        # Grid position
        ox = margin
        oy = max(margin, (scr_h - gph) // 2)

        # 2. Sidebar position & width:
        # Starts right after the grid + gap, and extends ALL the way to the right edge!
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
        self.astar_path = path
        self.astar_active_until = now + ASTAR_HINT_DURATION_SEC
        self.astar_cooldown_until = now + ASTAR_HINT_DURATION_SEC + ASTAR_HINT_COOLDOWN_SEC
        return True

    def get_astar_status(self) -> tuple[str, float]:
        """Returns ('ACTIVE', remaining_sec) or ('COOLDOWN', remaining_sec) or ('READY', 0.0)."""
        now = time.time()
        if now < self.astar_active_until:
            return "ACTIVE", max(0.0, self.astar_active_until - now)
        elif now < self.astar_cooldown_until:
            return "COOLDOWN", max(0.0, self.astar_cooldown_until - now)
        else:
            return "READY", 0.0

    def handle_events(self) -> tuple[bool, Optional[tuple[int, int]], bool, bool, bool]:
        """
        Polls user input in active gameplay.
        """
        human_action = None
        restart_triggered = False
        menu_triggered = False
        astar_requested = False

        mouse_pos = pygame.mouse.get_pos()
        self.restart_hovered = self.restart_btn_rect.collidepoint(mouse_pos)
        self.menu_btn_hovered = self.menu_btn_rect.collidepoint(mouse_pos)
        self.heatmap_toggle_hovered = self.heatmap_toggle_rect.collidepoint(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, None, False, False, False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.restart_hovered:
                        restart_triggered = True
                    elif self.menu_btn_hovered:
                        menu_triggered = True
                    elif self.heatmap_toggle_hovered:
                        self.show_heatmap = not self.show_heatmap
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.is_fullscreen:
                        self.toggle_fullscreen()
                    else:
                        menu_triggered = True
                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_r:
                    restart_triggered = True
                elif event.key == pygame.K_x:
                    astar_requested = True
                elif event.key == pygame.K_h:
                    self.show_heatmap = not self.show_heatmap
                elif event.key == pygame.K_g:
                    self.reveal_ghost = not self.reveal_ghost
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                # Directional Controls (Immediate response on initial keypress)
                elif event.key in (pygame.K_UP, pygame.K_w):
                    human_action = MOVES_DICT["UP"]
                    self.last_move_time = time.time() + 0.10  # Initial repeat debounce
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

        # Continuous automatic movement while holding directional keys
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

        return True, human_action, restart_triggered, menu_triggered, astar_requested

    def _draw_card(self, rect: pygame.Rect, bg_color: tuple[int, int, int], border_color: tuple[int, int, int]) -> None:
        """Draws rounded silver/slate container card."""
        pygame.draw.rect(self.screen, bg_color, rect, border_radius=10)
        pygame.draw.rect(self.screen, border_color, rect, width=1, border_radius=10)

    def render(self, belief_grid: Optional[np.ndarray] = None, mode_label: str = "Demo") -> None:
        """Renders grid canvas, cards, tokens, A* path overlay, and telemetry.
        Cell size and sidebar width are computed dynamically to FILL the actual
        screen resolution — works identically in windowed and fullscreen.
        """
        self.screen.fill(COLOR_WINDOW_BG)

        elapsed = time.time() - self.start_time
        pulse = 0.5 + 0.5 * math.sin(elapsed * 5.0)

        # Compute layout from live screen size (fills fullscreen automatically)
        cs, ox, oy, gpw, gph, sx, card_w = self._compute_layout()

        max_prob = float(np.max(belief_grid)) if belief_grid is not None else 0.0

        is_astar_active = time.time() < self.astar_active_until
        astar_cells = set(self.astar_path) if is_astar_active else set()

        # -------------------------------------------------------------
        # 1. Grid Canvas (Dark Charcoal interior with Silver frame)
        # -------------------------------------------------------------
        grid_frame = pygame.Rect(ox - 3, oy - 3, gpw + 6, gph + 6)
        pygame.draw.rect(self.screen, COLOR_GRID_BG, grid_frame, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_GRID_OUTER_FRAME, grid_frame, width=2, border_radius=12)

        for r in range(self.env.height):
            for c in range(self.env.width):
                rx = ox + c * cs
                ry = oy + r * cs
                cell_rect = pygame.Rect(rx, ry, cs, cs)

                if (r, c) in self.env.walls:
                    pygame.draw.rect(self.screen, COLOR_WALL, cell_rect)
                    inner_margin = max(2, cs // 10)
                    inner_w = cs - inner_margin * 2
                    inner_wall = pygame.Rect(rx + inner_margin, ry + inner_margin, inner_w, inner_w)
                    pygame.draw.rect(self.screen, COLOR_WALL_INNER, inner_wall, border_radius=4)
                    pygame.draw.rect(self.screen, COLOR_WALL_ACCENT, inner_wall, width=1, border_radius=4)
                    if cs >= 20:
                        pygame.draw.line(self.screen, COLOR_WALL_ACCENT,
                                         (rx + inner_margin + 3, ry + cs - inner_margin - 3),
                                         (rx + cs - inner_margin - 3, ry + inner_margin + 3), 1)
                        rivet_r = max(1, cs // 18)
                        pygame.draw.circle(self.screen, COLOR_WALL_RIVET, (rx + inner_margin + 4, ry + inner_margin + 4), rivet_r)
                        pygame.draw.circle(self.screen, COLOR_WALL_RIVET, (rx + cs - inner_margin - 4, ry + cs - inner_margin - 4), rivet_r)
                else:
                    if self.show_heatmap and belief_grid is not None:
                        prob = float(belief_grid[r, c])
                        tile_color = compute_gradient_color(prob, max_prob)
                    else:
                        tile_color = COLOR_GRID_SURFACE

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
        # 2. Hunter Token (Cyan Blue Disc)
        # -------------------------------------------------------------
        hr, hc = self.env.hunter_pos
        hx = ox + hc * cs + cs // 2
        hy = oy + hr * cs + cs // 2

        token_r = max(4, cs // 3)
        glow_rad = int(token_r * 1.25)
        pygame.draw.circle(self.screen, COLOR_HUNTER_RING, (hx, hy), glow_rad, 2)
        pygame.draw.circle(self.screen, COLOR_HUNTER, (hx, hy), token_r)
        pygame.draw.circle(self.screen, (255, 255, 255), (hx, hy), max(2, token_r // 3))

        # -------------------------------------------------------------
        # 3. Ghost Token (Invisible by Default; shown on capture or [G])
        # -------------------------------------------------------------
        if self.env.captured or self.reveal_ghost:
            gr, gc = self.env.true_ghost_pos
            gx = ox + gc * cs + cs // 2
            gy = oy + gr * cs + cs // 2
            ghost_token_r = max(4, cs // 3)
            ghost_rad = int(ghost_token_r * 1.3)
            pygame.draw.circle(self.screen, COLOR_GHOST_AURA, (gx, gy), ghost_rad, 2)
            pygame.draw.circle(self.screen, COLOR_GHOST, (gx, gy), ghost_token_r)
            if ghost_token_r >= 8:
                pygame.draw.circle(self.screen, (255, 255, 255), (gx - 4, gy - 2), 2)
                pygame.draw.circle(self.screen, (255, 255, 255), (gx + 4, gy - 2), 2)

        # -------------------------------------------------------------
        # 3b. Yellow Target Reticle — always tracks HMM MAP estimate
        # -------------------------------------------------------------
        if belief_grid is not None:
            raw_map = np.unravel_index(np.argmax(belief_grid), belief_grid.shape)
            tr, tc = int(raw_map[0]), int(raw_map[1])
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
        # 4. Dashboard Cards (Sidebar) — fills full screen height and width
        # -------------------------------------------------------------
        scr_h = self.screen.get_height()
        sy = oy

        # Distribute available vertical space across the 4 cards
        total_sidebar_h = gph          # match grid height exactly
        gap = 10
        total_gaps = gap * 3
        usable_h = total_sidebar_h - total_gaps

        # Fixed proportions (must sum to 1.0)
        c1_h = max(130, int(usable_h * 0.22))
        c2_h = max(120, int(usable_h * 0.20))
        c3_h = max(135, int(usable_h * 0.23))
        c4_h = usable_h - c1_h - c2_h - c3_h   # remainder goes to shortcuts

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
        status_text = "CAPTURED!" if self.env.captured else ("PAUSED" if self.paused else "ACTIVE")
        st_bg_color = (254, 202, 202) if self.env.captured else ((254, 240, 138) if self.paused else STATUS_ACTIVE_BG)
        st_txt_color = (185, 28, 28) if self.env.captured else ((133, 77, 14) if self.paused else STATUS_ACTIVE_TEXT)
        st_bg = pygame.Rect(sx + card_w - 95, sy + 38, 82, 24)
        pygame.draw.rect(self.screen, st_bg_color, st_bg, border_radius=6)
        st_surf = self.font_badge.render(status_text, True, st_txt_color)
        self.screen.blit(st_surf, (st_bg.centerx - st_surf.get_width() // 2, st_bg.centery - st_surf.get_height() // 2))

        # Restart & Menu Buttons
        btn_y = sy + 68
        self.restart_btn_rect = pygame.Rect(sx + 14, btn_y, (card_w - 34) // 2 + 10, 26)
        btn_bg = BTN_SECONDARY_HOVER if self.restart_hovered else BTN_SECONDARY_BG
        pygame.draw.rect(self.screen, btn_bg, self.restart_btn_rect, border_radius=6)
        btn_surf = self.font_badge.render("RESTART [R]", True, (255, 255, 255))
        self.screen.blit(btn_surf, (self.restart_btn_rect.centerx - btn_surf.get_width() // 2, self.restart_btn_rect.centery - btn_surf.get_height() // 2))

        self.menu_btn_rect = pygame.Rect(self.restart_btn_rect.right + 8, btn_y, card_w - self.restart_btn_rect.width - 36, 26)
        m_bg = (100, 116, 139) if self.menu_btn_hovered else (71, 85, 105)
        pygame.draw.rect(self.screen, m_bg, self.menu_btn_rect, border_radius=6)
        menu_surf = self.font_badge.render("MENU [Esc]", True, (255, 255, 255))
        self.screen.blit(menu_surf, (self.menu_btn_rect.centerx - menu_surf.get_width() // 2, self.menu_btn_rect.centery - menu_surf.get_height() // 2))

        # Heatmap toggle button
        self.heatmap_toggle_rect = pygame.Rect(sx + 14, sy + 102, card_w - 28, c1_h - 110)
        hm_status_text = "Heatmap: ON  [H]" if self.show_heatmap else "Heatmap: OFF  [H] to Enable"
        hm_btn_bg = (254, 240, 138) if self.show_heatmap else (226, 232, 240)
        hm_txt_col = (133, 77, 14) if self.show_heatmap else (71, 85, 105)
        pygame.draw.rect(self.screen, hm_btn_bg, self.heatmap_toggle_rect, border_radius=6)
        hm_surf = self.font_badge.render(hm_status_text, True, hm_txt_col)
        self.screen.blit(hm_surf, (self.heatmap_toggle_rect.centerx - hm_surf.get_width() // 2, self.heatmap_toggle_rect.centery - hm_surf.get_height() // 2))

        # --- Card 2: MISSION TELEMETRY ---
        sy += c1_h + gap
        card2_rect = pygame.Rect(sx, sy, card_w, c2_h)
        self._draw_card(card2_rect, CARD2_BG, CARD2_BORDER)
        t_lbl = self.font_subtitle.render("Mission Telemetry", True, CARD2_TITLE)
        self.screen.blit(t_lbl, (sx + 14, sy + 8))

        telemetry_rows = [
            ("Turn Step",       f"{self.env.turn} / {self.env.turn_limit}"),
            ("Sensor Reading",  f"{self.env.last_reading} units"),
            ("Sensor Noise (σ)", f"{self.env.sensor_sigma:.1f}"),
            ("Hunter Position", f"({int(self.env.hunter_pos[0])}, {int(self.env.hunter_pos[1])})")
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
        h_lbl = self.font_subtitle.render("HMM Belief State", True, CARD3_TITLE)
        self.screen.blit(h_lbl, (sx + 14, sy + 8))

        if belief_grid is not None:
            raw_map = np.unravel_index(np.argmax(belief_grid), belief_grid.shape)
            map_pos = f"({int(raw_map[0])}, {int(raw_map[1])})"
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

        radar_status, time_left = self.get_astar_status()
        radar_badge       = f"ACTIVE ({time_left:.1f}s)"    if radar_status == "ACTIVE"   else \
                            f"RECHARGING ({time_left:.1f}s)" if radar_status == "COOLDOWN" else "READY [X]"
        radar_badge_color = (234, 179, 8) if radar_status == "ACTIVE" else \
                            (185, 28, 28) if radar_status == "COOLDOWN" else (71, 85, 105)

        k_lbl = self.font_subtitle.render("Keyboard Controls", True, CARD4_TITLE)
        self.screen.blit(k_lbl, (sx + 14, sy + 8))

        radar_surf   = self.font_badge.render(radar_badge, True, (255, 255, 255))
        r_pill_rect  = pygame.Rect(sx + card_w - radar_surf.get_width() - 24, sy + 6, radar_surf.get_width() + 14, 20)
        pygame.draw.rect(self.screen, radar_badge_color, r_pill_rect, border_radius=6)
        self.screen.blit(radar_surf, (r_pill_rect.centerx - radar_surf.get_width() // 2, r_pill_rect.centery - radar_surf.get_height() // 2))

        shortcuts = [
            ("[X]",        "A* Ghost Radar (3s glow, 5s CD)"),
            ("[H]",        "Toggle Heatmap"),
            ("[G]",        "Reveal Ghost Position"),
            ("[R]",        "Restart & New Maze"),
            ("[F]",        "Toggle Fullscreen"),
            ("[Space]",    "Pause / Resume"),
            ("[W,A,S,D]",  "Move (Human Play Mode)")
        ]
        sc_step = max(17, (c4_h - 36) // len(shortcuts))
        sc_y = sy + 32
        for k, desc in shortcuts:
            k_surf = self.font_mono.render(k, True, CARD4_KEY)
            d_surf = self.font_body.render(desc, True, CARD4_DESC)
            self.screen.blit(k_surf, (sx + 14, sc_y))
            self.screen.blit(d_surf, (sx + 90, sc_y))
            sc_y += sc_step

        pygame.display.flip()

    # -------------------------------------------------------------------------
    # Screen 1: Start Menu (Clean Times New Roman, Gold & Slate Theme)
    # -------------------------------------------------------------------------
    def show_start_screen(self) -> str:
        """
        Renders Start Screen:
        - "GHOST HUNTER" Title (no square box, pure clean text)
        - Start, Main Menu, Exit buttons in Gold/Slate theme
        - No institutional FAST-NUCES branding
        """
        btn_w, btn_h = 320, 52

        while True:
            # Recompute center each frame (handles fullscreen resize)
            center_x = self.screen.get_width() // 2
            win_h = self.screen.get_height()

            btn_start = pygame.Rect(center_x - btn_w // 2, int(win_h * 0.42), btn_w, btn_h)
            btn_menu  = pygame.Rect(center_x - btn_w // 2, int(win_h * 0.42) + 70, btn_w, btn_h)
            btn_exit  = pygame.Rect(center_x - btn_w // 2, int(win_h * 0.42) + 140, btn_w, btn_h)

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

            # Recompute center for drawing (consistent with button positions above)
            center_x = self.screen.get_width() // 2
            win_h = self.screen.get_height()

            # Pure clean title without any boxes or artifacts
            hero_title = self.font_hero.render("GHOST HUNTER", True, (30, 41, 59))
            sub_hero = self.font_subtitle.render("Probabilistic State Estimation & Tracking with Hidden Markov Models", True, (100, 116, 139))

            self.screen.blit(hero_title, (center_x - hero_title.get_width() // 2, int(win_h * 0.18)))
            self.screen.blit(sub_hero, (center_x - sub_hero.get_width() // 2, int(win_h * 0.27)))

            # Button 1: START (Warm Gold)
            c_start = BTN_PRIMARY_HOVER if hover_start else BTN_PRIMARY_BG
            pygame.draw.rect(self.screen, c_start, btn_start, border_radius=8)
            txt_s = self.font_btn.render("START", True, (255, 255, 255))
            self.screen.blit(txt_s, (btn_start.centerx - txt_s.get_width() // 2, btn_start.centery - txt_s.get_height() // 2))

            # Button 2: MAIN MENU (Silver-Slate)
            c_menu = BTN_SECONDARY_HOVER if hover_menu else BTN_SECONDARY_BG
            pygame.draw.rect(self.screen, c_menu, btn_menu, border_radius=8)
            txt_m = self.font_btn.render("MAIN MENU", True, (255, 255, 255))
            self.screen.blit(txt_m, (btn_menu.centerx - txt_m.get_width() // 2, btn_menu.centery - txt_m.get_height() // 2))

            # Button 3: EXIT (Crimson)
            c_exit = BTN_DANGER_HOVER if hover_exit else BTN_DANGER_BG
            pygame.draw.rect(self.screen, c_exit, btn_exit, border_radius=8)
            txt_e = self.font_btn.render("EXIT", True, (255, 255, 255))
            self.screen.blit(txt_e, (btn_exit.centerx - txt_e.get_width() // 2, btn_exit.centery - txt_e.get_height() // 2))

            # Footer hint
            hint_fs = self.font_body.render("Press [F] to toggle Fullscreen", True, (148, 163, 184))
            footer = self.font_body.render("Click Start to choose difficulty · Click Main Menu for instructions · Click Exit to quit", True, (148, 163, 184))
            self.screen.blit(footer, (center_x - footer.get_width() // 2, win_h - 55))
            self.screen.blit(hint_fs, (center_x - hint_fs.get_width() // 2, win_h - 32))

            pygame.display.flip()
            self.clock.tick(30)

    # -------------------------------------------------------------------------
    # Screen 2: Main Menu Instructions & Summary Modal
    # -------------------------------------------------------------------------
    def show_instructions_screen(self) -> None:
        """
        Renders Main Menu instructions and game summary modal in Times New Roman.
        """
        while True:
            win_w = self.screen.get_width()
            win_h = self.screen.get_height()
            center_x = win_w // 2
            btn_back = pygame.Rect(center_x - 100, win_h - 65, 200, 42)

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
                    elif event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                        return

            self.screen.fill(COLOR_WINDOW_BG)

            t = self.font_title.render("MAIN MENU - GAME SUMMARY & INSTRUCTIONS", True, (30, 41, 59))
            self.screen.blit(t, (center_x - t.get_width() // 2, 25))

            # Card 1: What Does The Game Do?
            card1 = pygame.Rect(40, 70, win_w - 80, 135)
            self._draw_card(card1, CARD1_BG, CARD1_BORDER)
            s_title = self.font_subtitle.render("WHAT DOES THIS GAME DO?", True, (30, 41, 59))
            self.screen.blit(s_title, (55, 80))

            summary_lines = [
                "• Ghost Hunter is a probabilistic deduction game demonstrating AI tracking under sensor uncertainty.",
                "• An invisible ghost moves evasively through an obstacle maze every 1.0 second (real-time).",
                "• Because you cannot see the ghost, you only receive noisy distance readings corrupted by random error σ.",
                "• An AI Hidden Markov Model (HMM) maintains a 2D belief heatmap using the Forward Algorithm (Prediction & Correction).",
                "• The heatmap dynamically shows probability hotspots: Dark Navy (Low) -> Amber -> Neon Pink (Peak)."
            ]
            y = 104
            for line in summary_lines:
                txt = self.font_body.render(line, True, (51, 65, 85))
                self.screen.blit(txt, (55, y))
                y += 20

            # Card 2: How To Play
            card2 = pygame.Rect(40, 215, self.window_w - 80, 160)
            self._draw_card(card2, CARD2_BG, CARD2_BORDER)
            p_title = self.font_subtitle.render("HOW TO PLAY & WIN", True, (30, 41, 59))
            self.screen.blit(p_title, (55, 225))

            play_lines = [
                "• Objective: Track down the invisible ghost and step into its exact tile to capture it before the turn limit.",
                "• Hunter Controls: Use W, A, S, D or Arrow Keys to navigate through the unblocked paths.",
                "• Heatmap Toggle [H Key]: Press H anytime to toggle the real-time probability heatmap on or off.",
                "• Ghost Reveal [G Key]: Press G to toggle the debug reveal of the invisible ghost.",
                "• A* Ghost Radar [X Key]: Illuminates the exact shortest path to the ghost for 2-3 sec (5s cooldown).",
                "• Restart [R Key]: Generates a completely new random maze with new obstacle positions."
            ]
            y = 250
            for line in play_lines:
                txt = self.font_body.render(line, True, (51, 65, 85))
                self.screen.blit(txt, (55, y))
                y += 20

            # Card 3: Difficulties
            card3 = pygame.Rect(40, 385, self.window_w - 80, 125)
            self._draw_card(card3, CARD3_BG, CARD3_BORDER)
            d_title = self.font_subtitle.render("DIFFICULTY TIERS & GRID SIZES", True, CARD3_TITLE)
            self.screen.blit(d_title, (55, 395))

            diff_lines = [
                "• Beginner: 10x10 Grid (Turn Limit: 100) — Classic fast-paced tactical maze.",
                "• Intermediate: 15x15 Grid (Turn Limit: 150) — Wider maze with higher spatial ambiguity.",
                "• Hard: 20x20 Grid (Turn Limit: 200) — Dense labyrinth with active evasive ghost movement.",
                "• Extreme: 30x30 Grid (Turn Limit: 300) — Massive arena challenge for master AI hunters!"
            ]
            y = 420
            for line in diff_lines:
                txt = self.font_body.render(line, True, (113, 63, 18))
                self.screen.blit(txt, (55, y))
                y += 20

            # Back Button
            c_back = BTN_SECONDARY_HOVER if hover_back else BTN_SECONDARY_BG
            pygame.draw.rect(self.screen, c_back, btn_back, border_radius=8)
            b_txt = self.font_btn.render("BACK", True, (255, 255, 255))
            self.screen.blit(b_txt, (btn_back.centerx - b_txt.get_width() // 2, btn_back.centery - b_txt.get_height() // 2))

            pygame.display.flip()
            self.clock.tick(30)

    # -------------------------------------------------------------------------
    # Screen 3: Difficulty Selection Screen (Gold / Slate Theme)
    # -------------------------------------------------------------------------
    def show_difficulty_selection(self) -> Optional[str]:
        """
        Renders difficulty selector in Times New Roman.
        """
        card_w = 480
        card_h = 64
        diff_keys = ["Beginner", "Intermediate", "Hard", "Extreme"]

        while True:
            # Recompute layout every frame so it adapts to fullscreen
            center_x = self.screen.get_width() // 2
            win_h = self.screen.get_height()
            start_y = int(win_h * 0.26)

            btn_rects = []
            for i, key in enumerate(diff_keys):
                rect = pygame.Rect(center_x - card_w // 2, start_y + i * 82, card_w, card_h)
                btn_rects.append((key, rect))

            btn_back = pygame.Rect(center_x - 100, win_h - 65, 200, 42)
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

            t = self.font_hero.render("SELECT DIFFICULTY", True, (30, 41, 59))
            sub = self.font_subtitle.render("Choose grid scale to configure maze dimensions and search space", True, (100, 116, 139))
            self.screen.blit(t, (center_x - t.get_width() // 2, int(win_h * 0.10)))
            self.screen.blit(sub, (center_x - sub.get_width() // 2, int(win_h * 0.19)))

            colors_map = {
                "Beginner":     ((241, 245, 249), (203, 213, 225), (202, 138, 4)),
                "Intermediate": ((248, 250, 252), (203, 213, 225), (180, 83, 9)),
                "Hard":         ((254, 252, 232), (254, 240, 138), (161, 98, 7)),
                "Extreme":      ((254, 242, 242), (254, 202, 202), (185, 28, 28))
            }

            for key, rect in btn_rects:
                is_hovered = rect.collidepoint(mouse_pos)
                bg_col, border_col, accent_col = colors_map[key]
                if is_hovered:
                    bg_col = lerp_rgb(bg_col, (255, 255, 255), 0.5)
                self._draw_card(rect, bg_col, border_col)

                cfg = DIFFICULTY_CONFIGS[key]
                k_surf = self.font_btn.render(f"{key.upper()} ({cfg['width']}x{cfg['height']} Grid)", True, accent_col)
                self.screen.blit(k_surf, (rect.x + 18, rect.y + 12))

                d_surf = self.font_body.render(cfg['description'], True, (71, 85, 105))
                self.screen.blit(d_surf, (rect.x + 18, rect.y + 36))

                arrow_surf = self.font_btn.render("PLAY ->", True, accent_col if is_hovered else (148, 163, 184))
                self.screen.blit(arrow_surf, (rect.right - arrow_surf.get_width() - 18, rect.centery - arrow_surf.get_height() // 2))

            c_back = BTN_SECONDARY_HOVER if hover_back else BTN_SECONDARY_BG
            pygame.draw.rect(self.screen, c_back, btn_back, border_radius=8)
            b_txt = self.font_btn.render("BACK", True, (255, 255, 255))
            self.screen.blit(b_txt, (btn_back.centerx - b_txt.get_width() // 2, btn_back.centery - b_txt.get_height() // 2))

            hint = self.font_body.render("Press [F] to toggle Fullscreen  |  [1] Beginner  [2] Intermediate  [3] Hard  [4] Extreme", True, (148, 163, 184))
            self.screen.blit(hint, (center_x - hint.get_width() // 2, win_h - 32))

            pygame.display.flip()
            self.clock.tick(30)

    def close(self) -> None:
        """Closes Pygame window cleanly."""
        pygame.quit()
