"""Pygame menus, animated crypt board and HMM field instruments."""

from functools import lru_cache
from typing import Optional
import math
import time
import sys
import ctypes
import pygame
import numpy as np
from config import (
    ASTAR_HINT_DURATION_SEC,
    ASTAR_HINT_COOLDOWN_SEC,
    COLOR_WINDOW_BG,
    HUMAN_MOVE_INTERVAL_SEC,
    TAG_PILL_TEXT,
    MOVES_DICT,
)
from ghost_hunter.audio import GameAudio
from ghost_hunter.crypt_actors import CryptActors
from ghost_hunter.crypt_board import CryptBoard, belief_color, draw_map_seal, draw_radar_path
from ghost_hunter.crypt_theme import (
    BRASS,
    INK,
    IVORY,
    PARCHMENT,
    ROSE,
    VERDIGRIS,
    Ossuary,
    lettering,
    lerp_rgb,
    plaque,
    rule,
    sigil,
)
from ghost_hunter.environment import GridEnvironment
from ghost_hunter.mode_screens import CloseGame, choose_difficulty, instructions_screen, start_screen, window_closing
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

        self._display_size = None
        self._vignette_size = None
        self._fog_size = None
        self.ossuary = Ossuary()
        self.board = CryptBoard()
        self.actors = CryptActors()
        self.audio = GameAudio()
        self._text = lru_cache(maxsize=192)(self._render_text)
        self.show_probability = False
        self.blocked_until = 0
        self.blocked_hunter = ""
        self._sync_display()

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
        self.move_repeat_delay: float = HUMAN_MOVE_INTERVAL_SEC

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

    def _sync_display(self):
        self.screen = pygame.display.get_surface()
        size = self.screen.get_size()
        if size == self._display_size:
            return
        self._display_size = size
        width, height = size
        self.ui_scale = min(height / 1080, width / 1100)
        scale = self.ui_scale
        self.font_hero = lettering(122 * scale, "VigilBlackletter")
        self.font_chapter = lettering(86 * scale, "VigilBlackletter")
        self.font_title = lettering(35 * scale, "VigilBlackletter")
        self.font_btn = lettering(33 * scale, "VigilBlackletter")
        self.font_body = lettering(max(14, 23 * scale))
        self.font_mono = lettering(max(14, 23 * scale), "VigilFigures")
        self.font_badge = lettering(max(14, 24 * scale), "VigilFigures")
        self._titles = {}
        self._fallback = {}
        self._text.cache_clear()

    def _render_text(self, font, label, color, width=None, height=None):
        # Keep names in their original script when the engraved Latin face lacks it.
        if not label.isascii() and any(metric is None for metric in font.metrics(label)):
            key = font.get_height()
            if key not in self._fallback:
                self._fallback[key] = [
                    pygame.font.SysFont(family, max(10, round(key * .82)))
                    for family in ("segoeui", "nirmalaui", "microsoftyahei", "dejavusans")
                ]
            for candidate in self._fallback[key]:
                if all(metric is not None for metric in candidate.metrics(label)):
                    font = candidate
                    break
        ink = font.render(label, True, color)
        offset = max(1, round(self.ui_scale))
        surface = pygame.Surface((ink.get_width() + offset, ink.get_height() + offset), pygame.SRCALPHA)
        surface.blit(font.render(label, True, (5, 7, 8)), (offset, offset))
        surface.blit(ink, (0, 0))
        fit = min(1, max(1, width) / surface.get_width() if width is not None else 1,
                  max(1, height) / surface.get_height() if height is not None else 1)
        if fit < 1:
            return pygame.transform.smoothscale(surface, (
                max(1, round(surface.get_width() * fit)), max(1, round(surface.get_height() * fit))))
        return surface

    def _horror_title(self, label, color, width, height=None, *, hero=False):
        key = (label, color, width, height, hero)
        if key not in self._titles:
            self._titles[key] = self._text(self.font_hero if hero else self.font_chapter,
                                           label, color, width, height)
        return self._titles[key]

    def _compute_layout(self) -> tuple[int, int, int, int, int, int, int]:
        """Fill the height with whole grid cells and give the sidebar the remaining width."""
        width, height = self.screen.get_size()
        margin = max(6, round(14 * self.ui_scale))
        sidebar_min = min(round(370 * max(0.6, self.ui_scale)), width // 2)
        cs = max(1, min((height - 2 * margin) // self.env.height,
                        (width - sidebar_min - 3 * margin) // self.env.width))
        grid_w, grid_h = self.env.width * cs, self.env.height * cs
        sx = grid_w + 2 * margin
        return cs, margin, (height - grid_h) // 2, grid_w, grid_h, sx, width - sx - margin

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

    def wall_collision(self, name="Hunter"):
        self.blocked_hunter = name
        self.blocked_until = pygame.time.get_ticks() + 800
        self.audio.wall_hit()

    def trigger_astar_hint(self, path: list[tuple[int, int]]) -> bool:
        """Activates A* path hint for 3s if cooldown (5s) has expired."""
        now = time.time()
        if now < self.astar_cooldown_until:
            return False
        self.astar_use_count += 1
        self.astar_path = path
        self.astar_active_until = now + ASTAR_HINT_DURATION_SEC
        self.astar_cooldown_until = now + ASTAR_HINT_DURATION_SEC + ASTAR_HINT_COOLDOWN_SEC
        self.audio.sound_radar()
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

    def handle_events(self, events=None, *, competitive=False) -> tuple[bool, Optional[tuple[int, int]], bool, bool, bool, Optional[str]]:
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
        was_paused = self.paused

        mouse_pos = pygame.mouse.get_pos()
        self.restart_hovered = self.restart_btn_rect.collidepoint(mouse_pos)
        self.menu_btn_hovered = self.menu_btn_rect.collidepoint(mouse_pos)

        events = pygame.event.get() if events is None else events
        if window_closing(events):
            return False, None, False, False, False, None
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = event.pos
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
                        if self.restart_btn_rect.collidepoint(mouse_pos):
                            restart_triggered = True
                        elif self.menu_btn_rect.collidepoint(mouse_pos):
                            menu_triggered = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.paused:
                        self.paused = False
                    else:
                        menu_triggered = True
                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_F2:
                    self.audio.toggle_mute()
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif not self.paused:
                    if event.key == pygame.K_r:
                        restart_triggered = True
                    elif event.key == pygame.K_x:
                        astar_requested = True
                    elif event.key == pygame.K_h:
                        self.show_probability = not self.show_probability
                    elif competitive:
                        continue
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
        if not competitive and human_action is None and not self.paused:
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

        if was_paused != self.paused:
            self.audio.quiet(self.paused)
        if restart_triggered:
            self.audio.button("start")
        elif menu_triggered or pause_action == "quit_to_menu":
            self.audio.button("back")
        elif pause_action or was_paused != self.paused:
            self.audio.button()
        return True, human_action, restart_triggered, menu_triggered, astar_requested, pause_action

    def _instrument_panel(self, rect, accent):
        plaque(self.screen, rect, INK, lerp_rgb(INK, accent, .48), self.ui_scale)

    def _instrument_title(self, rect, title, accent, reserve=0):
        pad = max(6, round(14 * self.ui_scale))
        text = self._text(self.font_title, title, accent, rect.width - 2 * pad - reserve)
        self.screen.blit(text, (rect.left + pad, rect.top + pad))
        return pad, rect.top + pad + text.get_height() + max(4, pad // 2)

    def _draw_readout(self, rect, heading, rows, accent, *, reserve=0):
        self._instrument_panel(rect, accent)
        pad, top = self._instrument_title(rect, heading, accent, reserve)
        gap = max(3, pad // 2)
        width = (rect.width - 2 * pad - gap) // 2
        row_count = (len(rows) + 1) // 2
        height = max(1, (rect.bottom - pad - top - (row_count - 1) * gap) // max(1, row_count))
        for index, (label, value, level) in enumerate(rows):
            column = index % 2
            tile = pygame.Rect(rect.left + pad + column * (width + gap),
                               top + index // 2 * (height + gap), width, height)
            if index == len(rows) - 1 and column == 0:
                tile.width += width + gap
            wash = lerp_rgb((22, 25, 24), accent, 0.06)
            pygame.draw.rect(self.screen, wash, tile, border_radius=4)
            inset = max(4, pad // 2)
            if level is not None:
                rail = pygame.Rect(tile.left + inset, tile.bottom - 4, tile.width - 2 * inset, 2)
                pygame.draw.rect(self.screen, lerp_rgb(wash, accent, 0.28), rail)
                fill = rail.copy()
                fill.width = round(rail.width * max(0, min(1, level)))
                pygame.draw.rect(self.screen, accent, fill)
            label_font = self.font_mono if "[" in label else self.font_body
            name = self._text(label_font, label, PARCHMENT, (tile.width - 3 * inset) * 6 // 11)
            reading = self._text(self.font_mono, value, accent, (tile.width - 3 * inset) * 5 // 11)
            y = tile.centery - (2 if level is not None else 0)
            self.screen.blit(name, name.get_rect(midleft=(tile.left + inset, y)))
            self.screen.blit(reading, reading.get_rect(midright=(tile.right - inset, y)))

    def _draw_controls(self, rect, radar, accent, *, autonomous=False):
        self._instrument_panel(rect, accent)
        pad, top = self._instrument_title(rect, "Field Controls", accent)
        gap = max(3, pad // 2)
        width = (rect.width - 2 * pad - gap) // 2
        height = max(1, (rect.bottom - pad - top - 2 * gap) // 3)
        movement = ("H", "Switch Map") if autonomous else ("WASD", "Move / Arrows")
        entries = (("X", radar), ("P", "Pause"), ("G", "Reveal"),
                   ("R", "Restart"), ("F", "Fullscreen"), movement)
        for index, (key, label) in enumerate(entries):
            tile = pygame.Rect(rect.left + pad + index % 2 * (width + gap),
                               top + index // 2 * (height + gap), width, height)
            pygame.draw.rect(self.screen, (28, 31, 28), tile, border_radius=4)
            cap = self._text(self.font_mono, key, accent, width * 2 // 5 - gap)
            plate = cap.get_rect(midleft=(tile.left + gap * 2, tile.centery)).inflate(gap, 3)
            pygame.draw.rect(self.screen, (56, 49, 38), plate, border_radius=4)
            self.screen.blit(cap, cap.get_rect(center=plate.center))
            text = self._text(self.font_body, label, IVORY, tile.right - plate.right - 2 * gap)
            self.screen.blit(text, text.get_rect(midleft=(plate.right + gap, tile.centery)))

    def _draw_cobweb(self, ox: int, oy: int, size: int, corner: str = "TL") -> None:
        """Draws creepy cobweb motif in corner."""
        direction = 1 if corner == "TL" else -1
        angles = [i * math.pi / 14 for i in range(8)]
        for angle in angles:
            point = (ox + direction * size * math.cos(angle), oy + size * math.sin(angle))
            pygame.draw.aaline(self.screen, (65, 67, 65), (ox, oy), point)
        for index in range(1, 7):
            radius = size * index / 6
            points = [(ox + direction * radius * math.cos(angle), oy + radius * math.sin(angle))
                      for angle in angles]
            pygame.draw.aalines(self.screen, (54, 57, 54), False, points)

    def _menu_page(self, title, subtitle="", *, wide=False, hero=False, mood=None):
        """Compose the reading column against the shadowed side of the crypt."""
        self._sync_display()
        s = self.ui_scale
        self.ossuary.draw(self.screen, s, mood)
        w, h = self.screen.get_size()
        if w / h >= 1.35:
            left = round(w * .08)
            width = min(round(w * (.57 if wide else .46)), round((1100 if wide else 850) * s))
        else:
            left = round(w * .08)
            width = w - left * 2
        pane = pygame.Rect(left, round(h * .11), width, round(h * .79))
        header = self._horror_title(title, IVORY, width, round(h * (.15 if hero else .112)), hero=hero)
        self.screen.blit(header, (left, pane.top))
        y = pane.top + header.get_height() + round(7 * s)
        if subtitle:
            text = self._text(self.font_body, subtitle, PARCHMENT, width, round(h * .04))
            self.screen.blit(text, (left + round(2 * s), y))
            y += text.get_height() + round(15 * s)
        else:
            y += round(11 * s)
        rule(self.screen, left, y, min(width, round(168 * s)), BRASS)
        return pygame.Rect(left, y + round(24 * s), width, max(1, pane.bottom - y - round(24 * s)))

    def _menu_button(self, rect, label, hovered, accent=BRASS, *, motif=None,
                     hint=None, detail=None, primary=False):
        s = self.ui_scale
        focus = hovered or primary
        fill = lerp_rgb((17, 21, 21), accent, .18 if hovered else .105 if primary else .035)
        edge = lerp_rgb(fill, accent, .91 if focus else .53)
        plaque(self.screen, rect, fill, edge, s)
        pad = min(rect.width // 10, max(5, round(18 * s)))
        left, right, y = rect.left + pad, rect.right - pad, rect.centery
        if motif:
            radius = max(3, min(round(18 * s), rect.height // 4, rect.width // 15))
            center = (left + radius, y)
            sigil(self.screen, center, radius, motif, accent)
            left += 2 * radius + pad
            pygame.draw.line(self.screen, lerp_rgb(fill, edge, .65),
                             (left - pad//2, y - radius), (left - pad//2, y + radius))
            left += pad//2
        if hint:
            key = self._text(self.font_mono, hint, accent, max(1, rect.width // 5), rect.height // 2)
            self.screen.blit(key, key.get_rect(midright=(right, y)))
            right -= key.get_width() + pad
        width = max(1, right - left)
        title = self._text(self.font_btn, label, IVORY, width, round(rect.height * (.46 if detail else .72)))
        if detail:
            body = self._text(self.font_body, detail, PARCHMENT, width, round(rect.height * .34))
            gap = max(1, round(3 * s))
            top = y - (title.get_height() + gap + body.get_height()) // 2
            self.screen.blit(title, (left, top))
            self.screen.blit(body, (left, top + title.get_height() + gap))
        elif motif or hint:
            self.screen.blit(title, title.get_rect(midleft=(left, y)))
        else:
            self.screen.blit(title, title.get_rect(center=rect.center))

    def _draw_blood(self, rect: pygame.Rect) -> None:
        now = time.time()
        s = self.ui_scale
        pygame.draw.line(self.screen, (87, 45, 39), rect.topleft, (rect.right-1, rect.top))
        for i, fraction in enumerate((.06, .19, .21, .42, .59, .66, .84, .96)):
            x = rect.left + round(rect.width*fraction)
            cycle = (now*(.21+.025*(i%3)) + i*.47) % 1
            length = max(2, round((3+(4+i%4*3)*min(1, cycle*1.3))*s))
            thick = max(1, round((1.4+.6*(i%2))*s))
            y = rect.top + length
            pygame.draw.polygon(self.screen, (96, 42, 36),
                                ((x-2*thick, rect.top), (x+2*thick, rect.top), (x+thick, rect.top+2), (x, y)))
            if cycle > .72:
                tip = (x, y+round(48*s*(cycle-.72)**2))
                pygame.draw.circle(self.screen, (128, 58, 46), tip, thick)

    def _draw_vignette(self, rect: pygame.Rect) -> None:
        if self._vignette_size != rect.size:
            y, x = np.ogrid[-1:1:96j, -1:1:96j]
            alpha = np.uint8(np.clip((np.sqrt(x * x + y * y) - 0.66) * 75, 0, 55))
            mask = pygame.Surface((96, 96), pygame.SRCALPHA)
            mask.fill(COLOR_WINDOW_BG)
            pixels = pygame.surfarray.pixels_alpha(mask)
            pixels[:] = alpha.T
            del pixels
            self._vignette = pygame.transform.smoothscale(mask, rect.size)
            self._vignette_size = rect.size
        self.screen.blit(self._vignette, rect.topleft)

    def _draw_mist(self, maze: pygame.Rect, elapsed: float) -> None:
        size = (max(1, maze.width * 3 // 5), max(1, maze.height // 5))
        if size != self._fog_size:
            x, y = np.ogrid[-1:1:96j, -1:1:48j]
            taper = np.maximum(0, 1 - x * x - y * y) ** 2
            wisps = 0.58 + 0.32 * np.sin(16 * y + 4 * x) + 0.10 * np.sin(27 * y - 5 * x)
            cloud = pygame.Surface((96, 48), pygame.SRCALPHA)
            cloud.fill((158, 190, 175))
            alpha = pygame.surfarray.pixels_alpha(cloud)
            alpha[:] = np.uint8(taper * wisps * 30)
            del alpha
            self._fog = pygame.transform.smoothscale(cloud, size)
            self._fog_size = size
        for index in range(3):
            progress = (elapsed * (0.027 + 0.006 * index) + index / 3) % 1
            x = maze.left - size[0] + round(progress * (maze.width + size[0]))
            y = maze.top + round(maze.height * (0.09 + 0.29 * index + 0.022 * math.sin(elapsed + index)))
            self.screen.blit(self._fog, (x, y))

    def draw_pause_overlay(self):
        w, h = self.screen.get_size()
        shade = pygame.Surface((w, h))
        shade.fill((4, 8, 10))
        shade.set_alpha(192)
        self.screen.blit(shade, (0, 0))
        box = pygame.Rect(0, 0, min(round(w*.82), round(650*self.ui_scale)), round(h*.53))
        box.center = (w//2, h//2)
        plaque(self.screen, box, INK, BRASS, self.ui_scale)
        self._draw_blood(box.inflate(-max(4, round(28*self.ui_scale)), 0))
        pad = max(6, box.width//12)
        title = self._horror_title("Paused", IVORY, box.width-2*pad, box.height//4)
        self.screen.blit(title, title.get_rect(center=(box.centerx, box.top+box.height//5)))
        buttons = []
        for i, (label, motif, hint) in enumerate((("Resume", "lantern", "P"),
                ("Change Difficulty", "eye", ""), ("Exit to Main Menu", "door", ""))):
            rect = pygame.Rect(box.left+pad, box.top+round(box.height*(.37+i*.18)),
                               box.width-2*pad, round(box.height*.145))
            self._menu_button(rect, label, rect.collidepoint(pygame.mouse.get_pos()),
                              motif=motif, hint=hint, primary=i==0)
            buttons.append(rect)
        self.pause_resume_rect, self.pause_diff_rect, self.pause_quit_rect = buttons
        sound = self._text(self.font_mono, f"F2   Sound {self.audio.status}", PARCHMENT,
                           box.width-2*pad, max(1, box.height//17))
        self.screen.blit(sound, sound.get_rect(center=(box.centerx, box.bottom-box.height//17)))

    def render(self, belief_grid: Optional[np.ndarray] = None, mode_label: str = "Demo", *, race=None, player_name=None, save_error=None, active_time=None, autonomous=False) -> None:
        """Renders grid canvas, horror tokens, A* path overlay, and telemetry."""
        self._sync_display()
        self.screen.fill(COLOR_WINDOW_BG)

        elapsed = time.time() - self.start_time
        pulse = 0.5 + 0.5 * math.sin(elapsed * 5.0)

        cs, ox, oy, gpw, gph, sx, card_w = self._compute_layout()

        # Find MAP estimate (argmax of belief grid)
        map_cell: Optional[tuple[int, int]] = None
        if belief_grid is not None:
            raw_map = np.unravel_index(np.argmax(belief_grid), belief_grid.shape)
            map_cell = (int(raw_map[0]), int(raw_map[1]))

        probability = None
        if self.show_probability and map_cell is not None:
            peak = float(belief_grid[map_cell])
            probability = np.sqrt(np.clip(belief_grid / max(peak, 1e-15), 0, 1))
        searching = not self.env.captured and (race is None or not race[0].finished)
        maze_rect = pygame.Rect(ox, oy, gpw, gph)
        self.board.draw(self.screen, maze_rect, self.env, cs, map_cell if searching else None,
                        probability, elapsed)
        self._draw_vignette(maze_rect)
        self.screen.set_clip(maze_rect)
        self._draw_mist(maze_rect, elapsed)
        if searching and self.is_astar_active():
            draw_radar_path(self.screen, maze_rect, cs, self.astar_path, pulse)
        if searching and map_cell is not None:
            row, col = map_cell
            center = (ox + col * cs + cs // 2, oy + row * cs + cs // 2)
            draw_map_seal(self.screen, center, cs, pulse)

        positions = race[0].positions if race else [self.env.hunter_pos]
        shared = len(positions) == 2 and positions[0] == positions[1]
        for index, (row, col) in enumerate(positions):
            if self.env.captured and (row, col) == self.env.true_ghost_pos:
                continue
            cell = pygame.Rect(ox + col*cs, oy + row*cs, cs, cs)
            if shared:
                cell.width = cs//2
                cell.x += index * (cs-cell.width)
            self.actors.hunter(self.screen, cell, index, pulse)

        if self.env.captured or (searching and self.is_ghost_visible()):
            gr, gc = self.env.true_ghost_pos
            ghost_cell = pygame.Rect(ox + gc*cs, oy + gr*cs, cs, cs)
            if self.env.captured:
                captors = race[0].captors if race else (0,)
                self.actors.capture(self.screen, ghost_cell, maze_rect, captors, pulse)
            else:
                self.actors.wraith(self.screen, ghost_cell, elapsed)

        # -------------------------------------------------------------
        # 4. Sidebar Cards
        # -------------------------------------------------------------
        self.screen.set_clip(None)
        margin = ox
        gap = max(4, round(10 * self.ui_scale))
        usable_h = self.screen.get_height() - 2 * margin - 3 * gap
        heights = [round(usable_h * fraction) for fraction in (0.19, 0.25, 0.25)]
        heights.append(usable_h - sum(heights))
        cards = []
        y = margin
        for height in heights:
            cards.append(pygame.Rect(sx, y, card_w, height))
            y += height + gap

        header = cards[0]
        status = "CAPTURED" if self.env.captured else "PAUSED" if self.paused else "HUNTING"
        if race and race[0].finished and not self.env.captured:
            status = "TIME OUT"
        status_color = (255, 221, 133) if self.env.captured else (253, 224, 71) if self.paused else (187, 247, 208)
        self._instrument_panel(header, status_color)
        pad = max(5, round(14 * self.ui_scale))
        inner_w = card_w - 2 * pad
        title = self._horror_title("Ghost Hunter", IVORY, inner_w,
                                   round(header.height * 0.29))
        self.screen.blit(title, (header.left + pad, header.top + pad))

        tag_line = f"{player_name} / {mode_label}" if player_name else mode_label.upper()
        tag = self._text(self.font_badge, tag_line,
                         TAG_PILL_TEXT, inner_w * 2 // 3)
        badge = self._text(self.font_badge, status, status_color, inner_w // 3 - pad)
        tag_y = header.top + round(header.height * 0.48)
        self.screen.blit(tag, tag.get_rect(midleft=(header.left + pad, tag_y)))
        badge_rect = badge.get_rect(midright=(header.right - pad * 2, tag_y))
        plate = badge_rect.inflate(pad, max(4, pad // 2))
        pygame.draw.rect(self.screen, lerp_rgb((19, 18, 26), status_color, 0.13), plate, border_radius=5)
        self.screen.blit(badge, badge_rect)

        button_w = (inner_w - gap) // 2
        button_h = max(10, round(header.height * 0.25))
        button_y = header.bottom - pad - button_h
        self.restart_btn_rect = pygame.Rect(header.left + pad, button_y, button_w, button_h)
        self.menu_btn_rect = self.restart_btn_rect.move(button_w + gap, 0)
        self._menu_button(self.restart_btn_rect, "Restart", self.restart_hovered, ROSE, hint="R")
        self._menu_button(self.menu_btn_rect, "Menu", self.menu_btn_hovered, hint="Esc")

        cyan = VERDIGRIS
        rose = ROSE
        violet = BRASS
        turn_limit = self.env.turn_limit
        telemetry = [
            ("Turns", f"{self.env.turn} / {turn_limit}", self.env.turn / max(1, turn_limit)),
            ("Sensor", f"{self.env.last_reading} units", None),
            ("Noise σ", f"{self.env.sensor_sigma:.1f}", None),
            ("Hunter Cell", str(tuple(map(int, self.env.hunter_pos))), None),
            ("A* Uses", str(self.astar_use_count), None),
        ]
        if active_time is not None:
            telemetry.append(("Active Time", f"{active_time:.2f}s", None))
        if searching and pygame.time.get_ticks() < self.blocked_until:
            telemetry.append(("Wall hit", self.blocked_hunter, None))
        self._draw_readout(cards[1], "Mission Telemetry", telemetry, cyan)

        belief_rows = []
        if belief_grid is not None and map_cell is not None:
            positive = belief_grid[belief_grid > 0]
            entropy = float(-np.sum(positive * np.log2(positive)))
            confidence = float(np.max(belief_grid))
            revealed = self.reveal_ghost or self.env.captured
            distance = manhattan_distance(self.env.hunter_pos, self.env.true_ghost_pos) if revealed else None
            belief_rows = [
                ("Likely Cell", str(map_cell), None),
                ("Confidence", f"{confidence * 100:.1f}%", confidence),
                ("Entropy", f"{entropy:.2f} bits", None),
                ("True Distance", str(distance) if revealed else "Hidden", None),
                ("Ghost Cell", str(self.env.true_ghost_pos) if revealed else ("Hidden" if race else "Hidden [G]"), None),
                ("Map  [H]", "Probability" if self.show_probability else "Proximity", None),
            ]
        ramp_w = max(6, min(cards[2].width//4, round(120*self.ui_scale)))
        self._draw_readout(cards[2], "HMM Belief State", belief_rows, rose,
                           reserve=ramp_w+pad if probability is not None else 0)
        if probability is not None:
            # The same transfer function as the board, sampled at linear probabilities.
            bound = cards[2]
            ramp_h = max(2, round(5*self.ui_scale))
            ramp = pygame.Rect(bound.right-pad-ramp_w, bound.top+pad, ramp_w, ramp_h)
            for dx in range(ramp.width):
                value = (dx / max(1, ramp.width-1)) ** .5
                pygame.draw.line(self.screen, belief_color(value),
                                 (ramp.x+dx, ramp.top), (ramp.x+dx, ramp.bottom-1))
            caption = self._text(self.font_mono, f"0 – {peak:.1%}", PARCHMENT, ramp_w, 2*pad)
            self.screen.blit(caption, caption.get_rect(midtop=(ramp.centerx, ramp.bottom)))

        radar_status, remaining = self.get_astar_status()
        radar = "Radar Ready" if radar_status == "READY" else f"{radar_status.title()} {remaining:.1f}s"
        if race:
            round_state, names, elapsed, error, against_ai = race
            if against_ai:
                if round_state.finished:
                    outcome = round_state.scorecard().verdict
                elif round_state.capture_ms[0] is not None:
                    outcome = "You caught it. AI is still hunting."
                elif round_state.capture_ms[1] is not None:
                    outcome = "AI caught it. Your hunt continues."
                else:
                    outcome = "Highest capture score wins"
            elif len(round_state.winners) == 1:
                outcome = f"{names[round_state.winners[0]]} wins"
            elif round_state.winners:
                outcome = "Tie: " + " & ".join(names)
            else:
                outcome = "Turn limit reached" if round_state.finished else "First capture wins"
            self._draw_readout(cards[3], outcome, [
                ("Green cloak · WASD" if not against_ai else "Green cloak · WASD / Arrows", names[0], None),
                ("Red cloak · AI" if against_ai else "Red cloak · Arrows", names[1], None),
                ("Active Time", f"{elapsed:.2f}s", None),
                ("X · Radar", radar, None),
                ("Leaderboard" if error else "P · Pause", error or "F · Fullscreen", None),
            ], violet)
        else:
            self._draw_controls(cards[3], radar, violet, autonomous=autonomous)
            if save_error:
                warning = self._text(self.font_body, save_error, rose, inner_w)
                self.screen.blit(warning, warning.get_rect(midbottom=(header.centerx, button_y - 2)))

        # If game is paused, draw the dark horror overlay on top
        if self.paused:
            self.draw_pause_overlay()

        pygame.display.flip()

    def show_start_screen(self) -> str:
        self.audio.quiet(False)
        try:
            return start_screen(self)
        except CloseGame:
            return "exit"

    def show_instructions_screen(self):
        return instructions_screen(self)

    def show_difficulty_selection(self, *, on_close=None, demo=False):
        try:
            return choose_difficulty(self, demo=demo)
        except CloseGame:
            if on_close:
                on_close()
            raise

    def close(self) -> None:
        """Closes Pygame window."""
        try:
            self.audio.close()
        finally:
            pygame.quit()
