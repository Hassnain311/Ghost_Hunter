"""
Pygame Visualizer and Interactive GUI for Ghost Hunter.
Uses curated high-contrast color scheme ("Sea Escape" / "Deep Sapphire Radar")
with polished visual elements, distinct walls, crisp cards, and responsive restart.
"""

from typing import Optional
import math
import time
import pygame
import numpy as np
from config import (
    GRID_WIDTH,
    GRID_HEIGHT,
    COLOR_BG,
    COLOR_GRID_SURFACE,
    COLOR_GRID_LINE,
    COLOR_WALL,
    COLOR_WALL_ACCENT,
    COLOR_HUNTER,
    COLOR_HUNTER_GLOW,
    COLOR_GHOST,
    COLOR_GHOST_GLOW,
    COLOR_TARGET_MARKER,
    COLOR_PANEL_BG,
    COLOR_CARD_BG,
    COLOR_CARD_BORDER,
    COLOR_BTN_BG,
    COLOR_BTN_HOVER,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED,
    COLOR_ACCENT_CYAN,
    COLOR_SUCCESS,
    HEATMAP_P0,
    HEATMAP_P1,
    HEATMAP_P2,
    HEATMAP_P3,
    MOVES_DICT
)
from ghost_hunter.environment import GridEnvironment
from ghost_hunter.sensor_model import manhattan_distance


def lerp_rgb(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Smooth linear interpolation between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t)
    )


def compute_gradient_color(prob: float, max_prob: float) -> tuple[int, int, int]:
    """High-contrast 4-stage color gradient for belief heatmap."""
    if max_prob <= 1e-9:
        return HEATMAP_P0
    
    val = max(0.0, min(1.0, prob / max_prob))
    
    if val < 0.25:
        return lerp_rgb(HEATMAP_P0, HEATMAP_P1, val / 0.25)
    elif val < 0.65:
        return lerp_rgb(HEATMAP_P1, HEATMAP_P2, (val - 0.25) / 0.40)
    else:
        return lerp_rgb(HEATMAP_P2, HEATMAP_P3, (val - 0.65) / 0.35)


class PygameVisualizer:
    """Modern, high-contrast Pygame GUI for Ghost Hunter."""

    def __init__(self, env: GridEnvironment, cell_size: int = 60):
        pygame.init()
        self.env = env
        self.cell_size = cell_size
        self.grid_margin = 24
        
        self.grid_pixel_w = env.width * cell_size
        self.grid_pixel_h = env.height * cell_size
        
        self.sidebar_w = 340
        self.window_w = self.grid_margin * 2 + self.grid_pixel_w + self.sidebar_w
        self.window_h = self.grid_margin * 2 + self.grid_pixel_h
        
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption("Ghost Hunter · Probabilistic AI Tracking (HMM)")
        self.clock = pygame.time.Clock()
        
        # High legibility typography
        self.font_title = pygame.font.SysFont("Segoe UI, Inter, Arial", 22, bold=True)
        self.font_subtitle = pygame.font.SysFont("Segoe UI, Inter, Arial", 13, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI, Inter, Arial", 14)
        self.font_body_bold = pygame.font.SysFont("Segoe UI, Inter, Arial", 14, bold=True)
        self.font_mono = pygame.font.SysFont("Consolas, Courier New", 13)
        self.font_badge = pygame.font.SysFont("Segoe UI, Inter, Arial", 11, bold=True)
        self.font_btn = pygame.font.SysFont("Segoe UI, Inter, Arial", 13, bold=True)
        
        self.show_heatmap = True
        self.reveal_ghost = False
        self.paused = False
        self.start_time = time.time()
        
        # Restart button state
        self.restart_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.restart_hovered = False

    def handle_events(self) -> tuple[bool, Optional[tuple[int, int]], bool]:
        """Polls user input and click events."""
        human_action = None
        restart_triggered = False
        mouse_pos = pygame.mouse.get_pos()
        self.restart_hovered = self.restart_btn_rect.collidepoint(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, None, False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and self.restart_hovered:
                    restart_triggered = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, None, False
                elif event.key == pygame.K_r:
                    restart_triggered = True
                elif event.key == pygame.K_h:
                    self.show_heatmap = not self.show_heatmap
                elif event.key == pygame.K_g:
                    self.reveal_ghost = not self.reveal_ghost
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                
                # Directional Controls
                elif event.key in (pygame.K_UP, pygame.K_w):
                    human_action = MOVES_DICT["UP"]
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    human_action = MOVES_DICT["DOWN"]
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    human_action = MOVES_DICT["LEFT"]
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    human_action = MOVES_DICT["RIGHT"]
                elif event.key == pygame.K_PERIOD:
                    human_action = MOVES_DICT["STAY"]

        return True, human_action, restart_triggered

    def _draw_card(self, rect: pygame.Rect, border_color=COLOR_CARD_BORDER) -> None:
        """Draws rounded dashboard container card."""
        pygame.draw.rect(self.screen, COLOR_CARD_BG, rect, border_radius=10)
        pygame.draw.rect(self.screen, border_color, rect, width=1, border_radius=10)

    def render(self, belief_grid: Optional[np.ndarray] = None, mode_label: str = "Demo") -> None:
        """Renders grid canvas, cards, tokens, and telemetry."""
        self.screen.fill(COLOR_BG)
        
        elapsed = time.time() - self.start_time
        pulse = 0.5 + 0.5 * math.sin(elapsed * 5.0)
        
        ox = self.grid_margin
        oy = self.grid_margin
        
        max_prob = float(np.max(belief_grid)) if belief_grid is not None else 0.0

        # -------------------------------------------------------------
        # 1. Grid Canvas & Maze Cells
        # -------------------------------------------------------------
        grid_frame = pygame.Rect(ox - 3, oy - 3, self.grid_pixel_w + 6, self.grid_pixel_h + 6)
        pygame.draw.rect(self.screen, (14, 22, 38), grid_frame, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, grid_frame, width=2, border_radius=12)

        for r in range(self.env.height):
            for c in range(self.env.width):
                rx = ox + c * self.cell_size
                ry = oy + r * self.cell_size
                cell_rect = pygame.Rect(rx, ry, self.cell_size, self.cell_size)
                
                if (r, c) in self.env.walls:
                    # Clear, distinct blocked wall representation
                    pygame.draw.rect(self.screen, COLOR_WALL, cell_rect)
                    inner_wall = pygame.Rect(rx + 5, ry + 5, self.cell_size - 10, self.cell_size - 10)
                    pygame.draw.rect(self.screen, (12, 18, 30), inner_wall, border_radius=5)
                    pygame.draw.rect(self.screen, COLOR_WALL_ACCENT, inner_wall, width=1, border_radius=5)
                    # Diagonal hatch accent
                    pygame.draw.line(self.screen, COLOR_WALL_ACCENT, (rx + 10, ry + self.cell_size - 10), (rx + self.cell_size - 10, ry + 10), 1)
                else:
                    if self.show_heatmap and belief_grid is not None:
                        prob = float(belief_grid[r, c])
                        tile_color = compute_gradient_color(prob, max_prob)
                    else:
                        tile_color = COLOR_GRID_SURFACE
                        
                    pygame.draw.rect(self.screen, tile_color, cell_rect)
                    pygame.draw.rect(self.screen, COLOR_GRID_LINE, cell_rect, width=1)

        # -------------------------------------------------------------
        # 2. MAP Target Reticle
        # -------------------------------------------------------------
        if belief_grid is not None and not self.env.captured:
            map_r, map_c = np.unravel_index(np.argmax(belief_grid), belief_grid.shape)
            mx = ox + map_c * self.cell_size + self.cell_size // 2
            my = oy + map_r * self.cell_size + self.cell_size // 2
            
            rad = int(14 + 2 * pulse)
            pygame.draw.circle(self.screen, COLOR_TARGET_MARKER, (mx, my), rad, 2)
            pygame.draw.line(self.screen, COLOR_TARGET_MARKER, (mx - rad - 4, my), (mx - 6, my), 2)
            pygame.draw.line(self.screen, COLOR_TARGET_MARKER, (mx + 6, my), (mx + rad + 4, my), 2)
            pygame.draw.line(self.screen, COLOR_TARGET_MARKER, (mx, my - rad - 4), (mx, my - 6), 2)
            pygame.draw.line(self.screen, COLOR_TARGET_MARKER, (mx, my + 6), (mx, my + rad + 4), 2)

        # -------------------------------------------------------------
        # 3. Hunter Token (Aqua Wave with Arctic Aura)
        # -------------------------------------------------------------
        hr, hc = self.env.hunter_pos
        hx = ox + hc * self.cell_size + self.cell_size // 2
        hy = oy + hr * self.cell_size + self.cell_size // 2
        
        glow_rad = int(self.cell_size * 0.38 + 2 * pulse)
        pygame.draw.circle(self.screen, COLOR_HUNTER_GLOW, (hx, hy), glow_rad, 2)
        pygame.draw.circle(self.screen, COLOR_HUNTER, (hx, hy), self.cell_size // 3)
        pygame.draw.circle(self.screen, (255, 255, 255), (hx, hy), 4)

        # -------------------------------------------------------------
        # 4. Ghost Token (Coral Rose with Ethereal Aura)
        # -------------------------------------------------------------
        if self.env.captured or self.reveal_ghost:
            gr, gc = self.env.true_ghost_pos
            gx = ox + gc * self.cell_size + self.cell_size // 2
            gy = oy + gr * self.cell_size + self.cell_size // 2
            
            ghost_rad = int(self.cell_size * 0.40 + 3 * pulse)
            pygame.draw.circle(self.screen, COLOR_GHOST_GLOW, (gx, gy), ghost_rad, 2)
            pygame.draw.circle(self.screen, COLOR_GHOST, (gx, gy), self.cell_size // 3)
            # Eyes
            pygame.draw.circle(self.screen, (255, 255, 255), (gx - 5, gy - 3), 3)
            pygame.draw.circle(self.screen, (255, 255, 255), (gx + 5, gy - 3), 3)

        # -------------------------------------------------------------
        # 5. Dashboard Cards
        # -------------------------------------------------------------
        sx = ox + self.grid_pixel_w + 24
        sy = oy
        
        # Header Card
        header_card = pygame.Rect(sx, sy, self.sidebar_w - 24, 104)
        self._draw_card(header_card)
        
        title_surf = self.font_title.render("GHOST HUNTER", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title_surf, (sx + 16, sy + 12))
        
        tag_text = "AUTONOMOUS HMM DEMO" if mode_label == "Demo" else "INTERACTIVE PLAY"
        tag_bg = pygame.Rect(sx + 16, sy + 44, 155, 20)
        pygame.draw.rect(self.screen, (2, 132, 199), tag_bg, border_radius=6)
        tag_surf = self.font_badge.render(tag_text, True, (255, 255, 255))
        self.screen.blit(tag_surf, (sx + 22, sy + 47))
        
        status_text = "CAPTURED!" if self.env.captured else ("PAUSED" if self.paused else "ACTIVE")
        status_color = COLOR_SUCCESS if self.env.captured else ((255, 200, 30) if self.paused else COLOR_ACCENT_CYAN)
        status_surf = self.font_body_bold.render(status_text, True, status_color)
        self.screen.blit(status_surf, (sx + header_card.width - status_surf.get_width() - 16, sy + 45))

        # Restart Button
        btn_w = header_card.width - 32
        self.restart_btn_rect = pygame.Rect(sx + 16, sy + 70, btn_w, 24)
        btn_bg = COLOR_BTN_HOVER if self.restart_hovered else COLOR_BTN_BG
        pygame.draw.rect(self.screen, btn_bg, self.restart_btn_rect, border_radius=6)
        
        btn_txt = "⟳ RESTART & NEW MAZE [R]"
        btn_surf = self.font_btn.render(btn_txt, True, (255, 255, 255) if self.restart_hovered else (224, 242, 254))
        btn_x = self.restart_btn_rect.centerx - btn_surf.get_width() // 2
        btn_y = self.restart_btn_rect.centery - btn_surf.get_height() // 2
        self.screen.blit(btn_surf, (btn_x, btn_y))

        # Telemetry Card
        sy += 114
        telemetry_card = pygame.Rect(sx, sy, self.sidebar_w - 24, 142)
        self._draw_card(telemetry_card)
        
        card_lbl = self.font_subtitle.render("MISSION TELEMETRY", True, COLOR_ACCENT_CYAN)
        self.screen.blit(card_lbl, (sx + 16, sy + 10))
        
        turn_pct = min(1.0, self.env.turn / 100.0)
        metrics_rows = [
            ("Turn Step", f"{self.env.turn} / 100"),
            ("Sensor Reading", f"{self.env.last_reading} units"),
            ("Sensor Noise (σ)", f"{self.env.sensor_sigma:.1f}"),
            ("Hunter Position", f"({int(self.env.hunter_pos[0])}, {int(self.env.hunter_pos[1])})")
        ]
        
        row_y = sy + 32
        for lbl, val in metrics_rows:
            l_surf = self.font_body.render(lbl, True, COLOR_TEXT_SECONDARY)
            v_surf = self.font_mono.render(val, True, COLOR_TEXT_PRIMARY)
            self.screen.blit(l_surf, (sx + 16, row_y))
            self.screen.blit(v_surf, (sx + telemetry_card.width - v_surf.get_width() - 16, row_y))
            row_y += 22
            
        pbar_rect = pygame.Rect(sx + 16, sy + 126, telemetry_card.width - 32, 6)
        pygame.draw.rect(self.screen, (10, 17, 30), pbar_rect, border_radius=3)
        fill_w = int(pbar_rect.width * turn_pct)
        if fill_w > 0:
            fill_rect = pygame.Rect(sx + 16, sy + 126, fill_w, 6)
            bar_color = (244, 63, 94) if turn_pct > 0.8 else COLOR_HUNTER
            pygame.draw.rect(self.screen, bar_color, fill_rect, border_radius=3)

        # HMM Belief State Card
        sy += 152
        hmm_card = pygame.Rect(sx, sy, self.sidebar_w - 24, 150)
        self._draw_card(hmm_card)
        
        hmm_lbl = self.font_subtitle.render("HMM BELIEF STATE", True, COLOR_TARGET_MARKER)
        self.screen.blit(hmm_lbl, (sx + 16, sy + 10))
        
        if belief_grid is not None:
            raw_map = np.unravel_index(np.argmax(belief_grid), belief_grid.shape)
            map_pos = f"({int(raw_map[0])}, {int(raw_map[1])})"
            true_dist = manhattan_distance(self.env.hunter_pos, self.env.true_ghost_pos)
            map_err = manhattan_distance((int(raw_map[0]), int(raw_map[1])), self.env.true_ghost_pos)
            
            p = belief_grid[belief_grid > 0]
            entropy = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0
            
            hmm_rows = [
                ("MAP Estimate", map_pos),
                ("Peak Confidence", f"{max_prob * 100:.1f}%"),
                ("Belief Entropy", f"{entropy:.2f} bits"),
                ("True Distance", f"{true_dist if (self.reveal_ghost or self.env.captured) else 'Hidden'}"),
                ("Manhattan Error", f"{map_err if (self.reveal_ghost or self.env.captured) else 'Hidden'}")
            ]
            
            h_y = sy + 32
            for lbl, val in hmm_rows:
                l_surf = self.font_body.render(lbl, True, COLOR_TEXT_SECONDARY)
                v_surf = self.font_mono.render(val, True, (255, 255, 255) if "Peak" in lbl else COLOR_TEXT_PRIMARY)
                self.screen.blit(l_surf, (sx + 16, h_y))
                self.screen.blit(v_surf, (sx + hmm_card.width - v_surf.get_width() - 16, h_y))
                h_y += 21

        # Shortcuts Card
        sy += 160
        controls_card = pygame.Rect(sx, sy, self.sidebar_w - 24, 140)
        self._draw_card(controls_card)
        
        c_lbl = self.font_subtitle.render("KEYBOARD SHORTCUTS", True, COLOR_TEXT_MUTED)
        self.screen.blit(c_lbl, (sx + 16, sy + 10))
        
        shortcuts = [
            ("[R]", "Restart & Generate New Maze"),
            ("[H]", "Toggle Probability Heatmap"),
            ("[G]", "Debug Reveal Ghost"),
            ("[Space]", "Pause / Resume"),
            ("[W,A,S,D]", "Move (Human Play Mode)"),
            ("[ESC]", "Quit Game")
        ]
        
        sc_y = sy + 30
        for k, desc in shortcuts:
            k_surf = self.font_mono.render(k, True, COLOR_ACCENT_CYAN)
            d_surf = self.font_body.render(desc, True, COLOR_TEXT_SECONDARY)
            self.screen.blit(k_surf, (sx + 16, sc_y))
            self.screen.blit(d_surf, (sx + 88, sc_y))
            sc_y += 17

        pygame.display.flip()

    def close(self) -> None:
        """Closes Pygame window cleanly."""
        pygame.quit()
