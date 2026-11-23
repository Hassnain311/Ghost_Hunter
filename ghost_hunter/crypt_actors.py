"""Painted hunters, apparitions and the sealed capture on the crypt floor."""

from functools import lru_cache
import math

import pygame

from config import COLOR_GHOST_AURA, COLOR_HUNTER, COLOR_HUNTER_RING
from ghost_hunter.crypt_theme import ASSETS, BRASS, INK, ROSE, diamond, lerp_rgb, plaque


class CryptActors:
    def __init__(self):
        atlas = pygame.image.load(str(ASSETS / "crypt_actors.png")).convert_alpha()
        tile = atlas.get_width() // 2
        self._originals = {}
        for i, name in enumerate(("pilgrim", "warden", "wraith", "reliquary")):
            region = atlas.subsurface((i % 2 * tile, i // 2 * tile, tile, tile))
            cutout = region.subsurface(region.get_bounding_rect(min_alpha=3))
            self._originals[name] = cutout.premul_alpha()
        self._sprite = lru_cache(maxsize=64)(self._scale)

    def _scale(self, name, width, height):
        source = self._originals[name]
        bounds = source.get_rect().fit(0, 0, max(2, width), max(2, height))
        return pygame.transform.smoothscale(source, (max(1, bounds.width), max(1, bounds.height)))

    @staticmethod
    @lru_cache(maxsize=48)
    def _halo(radius, color):
        halo = pygame.Surface((2*radius+1, 2*radius+1), pygame.SRCALPHA)
        for ring in range(radius, 0, -1):
            opacity = round(122 * (1 - (ring-1)/radius)**2)
            pygame.draw.circle(halo, (*color, opacity), (radius, radius), ring)
        return halo

    def _light(self, screen, center, radius, color, intensity):
        glow = self._halo(max(2, round(radius)), color)
        glow.set_alpha(round(255 * intensity))
        screen.blit(glow, glow.get_rect(center=center))

    @staticmethod
    @lru_cache(maxsize=32)
    def _shadow(width, height, accent):
        shadow = pygame.Surface((width, height), pygame.SRCALPHA)
        rect = shadow.get_rect()
        for fraction, alpha in ((1, 38), (.86, 58), (.70, 100)):
            inner = rect.scale_by(fraction)
            pygame.draw.ellipse(shadow, (16, 19, 15, alpha), inner)
        pygame.draw.arc(shadow, (*accent, 186), rect, math.pi*1.18, math.pi*1.83,
                        max(1, height//10))
        return shadow

    def hunter(self, screen, cell, index, pulse):
        name, accent = (("pilgrim", COLOR_HUNTER), ("warden", ROSE))[index]
        image = self._sprite(name, round(cell.width*.92), round(cell.height*.90))
        body = image.get_rect(center=cell.center)
        shadow = self._shadow(max(3, round(body.width*.93)), max(3, round(body.height*.19)), accent)
        screen.blit(shadow, shadow.get_rect(midbottom=body.midbottom))
        lamp = (body.left + round(body.width*(.23 if index == 0 else .82)),
                body.top + round(body.height*.30))
        self._light(screen, lamp, body.height*.38, COLOR_HUNTER_RING, .72+.28*pulse)
        screen.blit(image, body, special_flags=pygame.BLEND_PREMULTIPLIED)

    def wraith(self, screen, cell, elapsed):
        image = self._sprite("wraith", round(cell.width*.90), round(cell.height*.86))
        sway = math.sin(elapsed*3.3)
        body = image.get_rect(center=(cell.centerx, cell.centery + round(cell.height*.04*sway)))
        self._light(screen, body.center, cell.width*.46, COLOR_GHOST_AURA, .64+.18*sway)
        screen.blit(image, body, special_flags=pygame.BLEND_PREMULTIPLIED)

    def reliquary(self, screen, bounds, pulse):
        image = self._sprite("reliquary", bounds.width, bounds.height)
        self._light(screen, bounds.center, bounds.height*.48, (255, 188, 85), .75+.25*pulse)
        screen.blit(image, image.get_rect(center=bounds.center), special_flags=pygame.BLEND_PREMULTIPLIED)

    def _captors(self, screen, box, winners, pulse):
        # The clasp and captive face stay in front. A tie has one hunter on each side.
        sides = (.24, .77) if len(winners) == 2 else (.225,)
        for index, x in zip(winners, sides):
            hunter = pygame.Rect(0, 0, round(box.width*.41), round(box.height*.76))
            hunter.center = (box.left + round(x*box.width), box.centery)
            self.hunter(screen, hunter, index, pulse)
        cage = pygame.Rect(0, 0, round(box.width*.52), round(box.height*.87))
        cage.center = (box.left + round(box.width*(.50 if len(winners)==2 else .71)), box.centery)
        self.reliquary(screen, cage, pulse)

    def capture(self, screen, cell, maze, winners, pulse):
        gold = lerp_rgb(BRASS, (255, 237, 169), pulse)
        frame = cell.inflate(-2, -2)
        cut = max(2, cell.width//9)
        for x, y, dx, dy in ((frame.left, frame.top, 1, 1), (frame.right-1, frame.top, -1, 1),
                             (frame.left, frame.bottom-1, 1, -1), (frame.right-1, frame.bottom-1, -1, -1)):
            pygame.draw.lines(screen, gold, False, [(x+dx*2*cut,y), (x+dx*cut,y),
                                                    (x,y+dy*cut), (x,y+dy*2*cut)], max(1, cell.width//36))
        detail = max(64, min(180, round(screen.get_height()*.104)))
        if cell.width >= detail*.76:
            self._captors(screen, frame, winners, pulse)
            return

        # At the smallest maze scales the cage would be only a few pixels wide.
        # Keep it on the actual tile and magnify the seal just above or below it.
        self.reliquary(screen, frame.inflate(-2, -2), pulse)
        gap = max(4, cell.height//5)
        inset = pygame.Rect(0, 0, min(maze.width, round(detail*1.23)), min(maze.height, detail))
        above = cell.top - gap - inset.height >= maze.top
        inset.centerx = cell.centerx
        inset.top = cell.top-gap-inset.height if above else cell.bottom+gap
        inset.clamp_ip(maze)
        origin = cell.midtop if above else cell.midbottom
        tip = (cell.centerx, inset.bottom-1 if above else inset.top)
        pygame.draw.line(screen, gold, origin, tip, max(1, detail//62))
        diamond(screen, origin, max(2, detail//30), gold)
        plaque(screen, inset, INK, gold, detail/125)
        self._captors(screen, inset.inflate(-4, -4), winners, pulse)
