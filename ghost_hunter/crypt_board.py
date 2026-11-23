"""Painted ossuary masonry and tracking overlays for the playable grid."""

import math

import numpy as np
import pygame

from config import (
    BELIEF_LOW, BELIEF_MID, BELIEF_PEAK, COLOR_ASTAR_PATH, COLOR_ASTAR_PATH_PULSE,
    COLOR_GRID_BG, COLOR_GRID_LINE, COLOR_GRID_OUTER_FRAME, COLOR_TARGET_MARKER,
    HEATMAP_DIST_1, HEATMAP_DIST_2_3, HEATMAP_DIST_4_5,
)
from ghost_hunter.crypt_theme import ASSETS, BRASS, IVORY, diamond, lerp_rgb


def belief_color(strength):
    """Shared by the stone lighting and the probability legend."""
    if strength <= .55:
        return lerp_rgb(BELIEF_LOW, BELIEF_MID, strength / .55)
    return lerp_rgb(BELIEF_MID, BELIEF_PEAK, (strength - .55) / .45)


def _light_filter(stone, color):
    """SDL multiply/add factors that shift the stone's midtone while retaining relief."""
    multiply = tuple(min(255, round(255 * target / base)) for base, target in zip(stone, color))
    add = tuple(max(0, target - base) for base, target in zip(stone, color))
    return multiply, add


def _engrave_floor(surface):
    """Inlay a single eight-petal cathedral rosette beneath the joints and tombs."""
    width, height = surface.get_size()
    radius = round(min(width, height) * .34)
    if radius < 12:
        return
    margin = 3
    center = radius + margin
    plate = pygame.Surface((2 * center + 1, 2 * center + 1), pygame.SRCALPHA)
    stroke = max(1, radius // 180)
    dark, gold = (27, 31, 25, 146), (194, 169, 112, 94)
    for fraction in (.15, .83, .87, .98):
        r = round(radius * fraction)
        pygame.draw.circle(plate, dark, (center, center), r, stroke + 1)
        pygame.draw.circle(plate, gold, (center, center), r - stroke, stroke)

    def point(angle, distance):
        return (center + round(radius * distance * math.cos(angle)),
                center + round(radius * distance * math.sin(angle)))

    for petal in range(8):
        axis = petal * math.tau / 8
        sides = []
        for sign in (-1, 1):
            side = []
            for step in range(25):
                t = step / 24
                angle = axis + sign * .30 * math.sin(math.pi * t)
                side.append(point(angle, .16 + .66 * t))
            sides.append(side)
        leaf = sides[0] + sides[1][-2:0:-1]
        pygame.draw.lines(plate, dark, True, leaf, stroke + 1)
        pygame.draw.aalines(plate, gold, True, leaf)
        for offset in (-.09, 0, .09):
            angle = axis + offset
            pygame.draw.line(plate, gold, point(angle, .89), point(angle, .955), stroke)
    surface.blit(plate, plate.get_rect(center=(width // 2, height // 2)))


def _candle_cells(walls, rows, columns, rng):
    """A few widely spaced wall altars provide warm landmarks on every maze size."""
    candidates = sorted(walls)
    limit = max(1, min(6, round(len(walls) / 18)))
    spacing = max(3, min(rows, columns) // 4)
    chosen = []
    for index in rng.permutation(len(candidates)):
        r, c = candidates[index]
        if all(abs(r - y) + abs(c - x) >= spacing for y, x in chosen):
            chosen.append((r, c))
            if len(chosen) == limit:
                break
    return chosen


class CryptBoard:
    def __init__(self):
        sheet = pygame.image.load(ASSETS / "crypt_tiles.jpg").convert()
        side = sheet.get_width() // 3
        self.tiles = [sheet.subsurface((col * side, row * side, side, side))
                      for row in range(2) for col in range(3)]
        self.stone_colors = [pygame.transform.average_color(tile)[:3] for tile in self.tiles[:3]]
        self.ring_colors = (HEATMAP_DIST_1, HEATMAP_DIST_2_3, HEATMAP_DIST_4_5)
        self.ring_filters = [
            [_light_filter(stone, lerp_rgb(stone, color, .65)) for color in self.ring_colors]
            for stone in self.stone_colors
        ]
        self.belief_filters = [
            [_light_filter(stone, belief_color(i / 255)) for i in range(256)]
            for stone in self.stone_colors
        ]
        self.layout = None

    def _build(self, rows, columns, cell, walls):
        rng = np.random.default_rng(503)  # Visual variation is independent of the simulation.
        scaled = [pygame.transform.smoothscale(tile, (cell, cell)) for tile in self.tiles]
        paving = [[pygame.transform.rotate(tile, angle) for angle in (0, 90, 180, 270)]
                  for tile in scaled[:3]]
        styles = rng.integers(0, 12, (rows, columns))
        self.base = pygame.Surface((columns * cell, rows * cell)).convert()
        self.open_tiles = []
        for r, c in np.ndindex(rows, columns):
            material, rotation = divmod(int(styles[r, c]), 4)
            rect = pygame.Rect(c * cell, r * cell, cell, cell)
            self.base.blit(paving[material][rotation], rect)
            if (r, c) not in walls:
                self.open_tiles.append((r, c, rect, material))
        _engrave_floor(self.base)
        w, h = self.base.get_size()
        for x in range(0, w, cell):
            pygame.draw.line(self.base, COLOR_GRID_LINE, (x, 0), (x, h))
        for y in range(0, h, cell):
            pygame.draw.line(self.base, COLOR_GRID_LINE, (0, y), (w, y))

        # Occlusion falls onto the adjacent paving, then the raised stone covers its own shadow.
        shadow = pygame.Surface((cell, cell), pygame.SRCALPHA)
        for inset, alpha in ((0, 28), (1, 55), (2, 85)):
            shadow_rect = shadow.get_rect().inflate(-2 * inset, -2 * inset)
            if shadow_rect.width > 0:
                pygame.draw.rect(shadow, (13, 16, 14, alpha), shadow_rect, border_radius=max(1, cell // 11))
        for r, c in walls:
            self.base.blit(shadow, (c * cell + cell // 12, r * cell + cell // 8))
        self.candles = _candle_cells(walls, rows, columns, rng)
        for r, c in walls:
            variant = 2 if (r, c) in self.candles else int(styles[r, c]) % 2
            self.base.blit(scaled[3 + variant], (c * cell, r * cell))

        x, y = np.ogrid[-1:1:64j, -1:1:64j]
        # Two soft shafts of clerestory light cross the paving and carved stone together.
        beam = x + .68 * y + .24
        wash = pygame.Surface((64, 64), pygame.SRCALPHA)
        wash.fill((149, 182, 163))
        alpha = pygame.surfarray.pixels_alpha(wash)
        alpha[:] = np.uint8(32 * np.exp(-(beam / .24) ** 2) + 16 * np.exp(-((beam - .44) / .13) ** 2))
        del alpha
        self.base.blit(pygame.transform.smoothscale(wash, self.base.get_size()), (0, 0))

        lamp = pygame.Surface((64, 64), pygame.SRCALPHA)
        lamp.fill((255, 171, 77))
        alpha = pygame.surfarray.pixels_alpha(lamp)
        alpha[:] = np.uint8(97 * np.maximum(0, 1 - x*x - y*y) ** 3)
        del alpha
        self.glow = pygame.transform.smoothscale(lamp, (2 * cell, 2 * cell))
        self.field = self.base.copy()
        self.signal = None

    def _paint(self, map_cell, levels):
        signal = levels.tobytes() if levels is not None else map_cell
        if signal == self.signal:
            return
        self.signal = signal
        self.field.blit(self.base, (0, 0))
        if signal is None:
            return
        for r, c, rect, material in self.open_tiles:
            if levels is not None:
                multiply, add = self.belief_filters[material][levels[r, c]]
            else:
                dr, dc = r - map_cell[0], c - map_cell[1]
                distance = abs(dr) + abs(dc)
                if distance == 0 or distance > 5:
                    continue
                band = 0 if distance == 1 else 1 if distance <= 3 else 2
                multiply, add = self.ring_filters[material][band]
            if multiply != (255, 255, 255):
                self.field.fill(multiply, rect, special_flags=pygame.BLEND_RGB_MULT)
            if add != (0, 0, 0):
                self.field.fill(add, rect, special_flags=pygame.BLEND_RGB_ADD)
            if levels is None and distance in (1, 3, 5) and rect.width >= 10:
                # Illuminate only the outward edge of each band; joints still read as stone.
                inset = max(1, rect.width // 17)
                inner = rect.inflate(-2 * inset, -2 * inset)
                shade = lerp_rgb(self.ring_colors[band], IVORY, .18)
                stroke = max(1, rect.width // 58)
                if dr <= 0:
                    pygame.draw.line(self.field, shade, inner.topleft, inner.topright, stroke)
                if dr >= 0:
                    pygame.draw.line(self.field, shade, inner.bottomleft, inner.bottomright, stroke)
                if dc <= 0:
                    pygame.draw.line(self.field, shade, inner.topleft, inner.bottomleft, stroke)
                if dc >= 0:
                    pygame.draw.line(self.field, shade, inner.topright, inner.bottomright, stroke)

    def draw(self, screen, bounds, env, cell, map_cell, probability, elapsed):
        key = (env.height, env.width, cell, frozenset(env.walls))
        if key != self.layout:
            self._build(*key)
            self.layout = key
        levels = None if probability is None else np.rint(probability * 255).astype(np.uint8)
        self._paint(map_cell, levels)
        rim = max(2, min(bounds.left, bounds.top, max(3, round(bounds.height / 150))))
        outer = bounds.inflate(2 * rim, 2 * rim)
        pygame.draw.rect(screen, COLOR_GRID_BG, outer, rim)
        pygame.draw.rect(screen, COLOR_GRID_OUTER_FRAME, outer, 1)
        pygame.draw.rect(screen, (102, 91, 65), bounds.inflate(2, 2), 1)
        reach = max(4, rim * 5)
        for x, dx in ((outer.left, 1), (outer.right - 1, -1)):
            for y, dy in ((outer.top, 1), (outer.bottom - 1, -1)):
                points = [(x + dx * reach, y + dy * 2), (x + dx * 3, y + dy * 2),
                          (x + dx * 2, y + dy * reach)]
                pygame.draw.lines(screen, BRASS, False, points, max(1, rim // 4))
                diamond(screen, (x + dx * rim // 2, y + dy * rim // 2), max(1, rim // 3), IVORY)
        screen.blit(self.field, bounds)
        previous_clip = screen.get_clip()
        screen.set_clip(bounds)
        for i, (r, c) in enumerate(self.candles):
            flame = (bounds.left + round((c + .50) * cell), bounds.top + round((r + .21) * cell))
            self.glow.set_alpha(round(94 + 18 * math.sin(elapsed * 3.3 + i) + 9 * math.sin(elapsed * 6.1 + i)))
            screen.blit(self.glow, self.glow.get_rect(center=flame))
        screen.set_clip(previous_clip)


def draw_map_seal(screen, center, cell, pulse):
    """An open surveyor's sight; its four points leave room for a hunter on the MAP."""
    x, y = center
    radius = max(2, round(cell * .38))
    weight = max(1, cell // 40)
    ink = (37, 26, 22)
    for color, thickness in ((ink, weight + 2), (BRASS, weight)):
        pygame.draw.circle(screen, color, center, radius, min(radius, thickness))
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        mx, my = x + dx * radius, y + dy * radius
        wing = max(2, round(cell * .07))
        points = [(mx + dx*wing, my + dy*wing), (mx - dy*wing, my + dx*wing),
                  (mx - dx*wing*2, my - dy*wing*2), (mx + dy*wing, my - dx*wing)]
        pygame.draw.polygon(screen, lerp_rgb(COLOR_TARGET_MARKER, IVORY, .32 + .25*pulse), points)
        pygame.draw.aalines(screen, ink, True, points)


def draw_radar_path(screen, bounds, cell, path, pulse):
    if not path:
        return
    points = [(bounds.left + c * cell + cell // 2, bounds.top + r * cell + cell // 2) for r, c in path]
    ember = lerp_rgb(lerp_rgb(COLOR_ASTAR_PATH, COLOR_ASTAR_PATH_PULSE, pulse), IVORY, .26)
    for i, (a, b) in enumerate(zip(points, points[1:])):
        # Two-player radar can concatenate two routes ending at the same ghost.
        if path[i] != path[-1] and abs(a[0]-b[0]) + abs(a[1]-b[1]) == cell:
            pygame.draw.line(screen, (79, 48, 28), a, b, max(3, cell // 13))
            pygame.draw.line(screen, ember, a, b, max(1, cell // 22))
    for point in points:
        diamond(screen, point, max(2, cell // 13), (72, 44, 26))
        diamond(screen, point, max(1, cell // 18), IVORY)
