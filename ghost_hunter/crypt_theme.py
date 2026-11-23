"""Offline lettering and artwork for the crypt menus and field instruments."""

import math
from pathlib import Path

import numpy as np
import pygame


ASSETS = Path(__file__).with_name("assets")
IVORY = (251, 235, 206)
PARCHMENT = (224, 211, 181)
BRASS = (226, 181, 111)
VERDIGRIS = (153, 218, 199)
ROSE = (244, 153, 153)
INK = (16, 19, 18)
RULE = (119, 95, 65)


def lerp_rgb(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Smooth linear interpolation between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t)
    )


def lettering(pixels, face="VigilText"):
    return pygame.font.Font(ASSETS / (face + ".ttf"), max(10, round(pixels)))


def diamond(screen, center, r, color, width=0):
    x, y = center
    pygame.draw.polygon(screen, color, [(x, y-r), (x+r, y), (x, y+r), (x-r, y)], width)


def rule(screen, x, y, width, color=RULE):
    pygame.draw.line(screen, color, (x, y), (x + width, y))
    diamond(screen, (x + width, y), 3, color)


def sigil(screen, center, radius, motif, color):
    """Small engraved emblems use one stroke weight across menu scales."""
    x, y = center
    stroke = max(1, round(radius / 10))

    def line(points, closed=False):
        scaled = [(round(x + a * radius), round(y + b * radius)) for a, b in points]
        pygame.draw.lines(screen, color, closed, scaled, stroke)
        pygame.draw.aalines(screen, color, closed, scaled)

    if motif == "lantern":
        line([(-.22,-.54),(-.22,-.83),(0,-1),(.22,-.83),(.22,-.54)])
        line([(-.56,-.43),(0,-.68),(.56,-.43),(.4,.52),(0,.83),(-.4,.52)], True)
        line([(-.5,-.36),(.5,-.36)])
        line([(-.35,.49),(.35,.49)])
        diamond(screen, (x, y + radius//9), max(2, radius//4), color)
    elif motif == "book":
        line([(-.88,-.63),(-.36,-.67),(0,-.41),(.36,-.67),(.88,-.63),(.87,.52),
              (.35,.48),(0,.75),(-.35,.48),(-.87,.52)], True)
        line([(0,-.4),(0,.7)])
        line([(-.69,-.22),(-.37,-.21),(-.18,-.08)])
        line([(.18,-.08),(.37,-.21),(.69,-.22)])
    elif motif == "hourglass":
        line([(-.64,-.76),(.64,-.76)])
        line([(-.51,-.66),(-.38,-.35),(.4,.39),(.48,.64),(-.48,.64),(-.37,.37),
              (.38,-.35),(.51,-.66)], True)
        line([(-.64,.78),(.64,.78)])
    elif motif == "duel":
        for offset in (-.38, .38):
            line([(offset-.23,-.13),(offset,-.68),(offset+.23,-.13),(offset,.1)], True)
            line([(offset-.39,.62),(offset-.21,.21),(offset+.21,.21),(offset+.39,.62)])
    elif motif == "eye":
        line([(-.91,0),(-.45,-.42),(0,-.52),(.45,-.42),(.91,0),(.45,.42),
              (0,.52),(-.45,.42)], True)
        diamond(screen, (x, y), max(2, radius//3), color)
        line([(0,-.98),(0,-.68)])
        line([(0,.68),(0,.98)])
    elif motif == "door":
        line([(-.57,.69),(-.57,-.31),(0,-.9),(.57,-.31),(.57,.69)])
        line([(-.22,.68),(-.22,-.21),(.13,-.48),(.13,.68)])
        line([(-.79,.72),(.79,.72)])


def plaque(screen, rect, fill, edge, scale):
    """Ink panel edged in brass with clipped corners and inset engraving."""
    x, y, w, h = rect
    cut = max(2, min(round(10 * scale), w // 8, h // 5))
    r, b = x + w - 1, y + h - 1
    points = [(x+cut,y),(r-cut,y),(r,y+cut),(r,b-cut),(r-cut,b),(x+cut,b),(x,b-cut),(x,y+cut)]
    pygame.draw.polygon(screen, fill, points)
    pygame.draw.aalines(screen, edge, True, points)
    pygame.draw.line(screen, edge, (x+cut, y+3), (x+min(w//4, 4*cut), y+3))
    pygame.draw.line(screen, edge, (r-min(w//4, 4*cut), b-3), (r-cut, b-3))


class Ossuary:
    """Cache the painted scene and animate candlelight and low-lying mist over it."""

    def __init__(self):
        self.art = pygame.image.load(ASSETS / "ossuary.jpg").convert()
        self.backdrop = None

    def draw(self, screen, scale, mood=None):
        w, h = screen.get_size()
        if self.backdrop is None or self.backdrop.get_size() != (w, h):
            sw, sh = self.art.get_size()
            view = pygame.Rect(0, 0, w, h).fit(self.art.get_rect())
            view.center = (sw // 2, sh // 2)
            self.backdrop = pygame.transform.smoothscale(self.art.subsurface(view), (w, h))
            # A reading area on the left and light falloff at the edges.
            x, y = np.ogrid[0:1:160j, 0:1:90j]
            left = np.exp(-((x - .25) / .40) ** 4)
            edge = np.minimum(1, 1.5 * abs(y - .5))
            alpha = np.clip(20 + 77 * left + 88 * edge ** 3, 0, 190)
            mask = pygame.Surface(alpha.shape, pygame.SRCALPHA)
            mask.fill((5, 9, 11))
            pixels = pygame.surfarray.pixels_alpha(mask)
            pixels[:] = alpha.astype(np.uint8)
            del pixels
            self.backdrop.blit(pygame.transform.smoothscale(mask, (w, h)), (0, 0))
            # Anchor light to the candles in the painting, including on cropped screens.
            self.lamps = [
                (round((sw*u-view.left)*w/view.width), round((sh*v-view.top)*h/view.height))
                for u, v in ((.045, .795), (.153, .785))
            ]
            lx, ly = np.ogrid[-1:1:64j, -1:1:64j]
            lamp = pygame.Surface((64, 64), pygame.SRCALPHA)
            lamp.fill((236, 153, 68))
            fade = pygame.surfarray.pixels_alpha(lamp)
            fade[:] = np.uint8(120 * np.maximum(0, 1 - lx*lx - ly*ly) ** 3)
            del fade
            side = max(8, round(.14*h))
            self.lamp = pygame.transform.smoothscale(lamp, (side, side))
            fx, fy = np.ogrid[-1:1:180j, -1:1:50j]
            strand = np.exp(-((fy-.18*np.sin(5*fx))/.34)**2) * np.maximum(0, 1-fx*fx)**2
            self.mist = {}
            for key, color in ((None, (148, 198, 185)), ("capture", (226, 190, 133)), ("escape", (169, 109, 121))):
                veil = pygame.Surface(strand.shape, pygame.SRCALPHA)
                veil.fill(color)
                alpha_view = pygame.surfarray.pixels_alpha(veil)
                alpha_view[:] = np.uint8(49*strand)
                del alpha_view
                self.mist[key] = pygame.transform.smoothscale(veil, (max(1,round(.50*w)), max(1,round(.15*h))))
            m = max(5, round(18 * scale))
            frame = pygame.Rect(m, m, w - 2*m, h - 2*m)
            pygame.draw.rect(self.backdrop, (81, 75, 54), frame, 1)
            reach = max(8, round(44 * scale))
            for cx, dx in ((frame.left, 1), (frame.right-1, -1)):
                for cy, dy in ((frame.top, 1), (frame.bottom-1, -1)):
                    pygame.draw.lines(self.backdrop, RULE, False, [(cx+dx*reach,cy+dy*4),
                        (cx+dx*4,cy+dy*4),(cx+dx*4,cy+dy*reach)])
                    diamond(self.backdrop, (cx+dx*9,cy+dy*9), max(2, m//5), BRASS)
        screen.blit(self.backdrop, (0, 0))
        seconds = pygame.time.get_ticks() / 1000
        for i, pos in enumerate(self.lamps):
            strength = round(47 + 13*math.sin(3*seconds+i) + 7*math.sin(7.1*seconds+2*i))
            self.lamp.set_alpha(strength)
            screen.blit(self.lamp, self.lamp.get_rect(center=pos))
        veil = self.mist[mood]
        for i in range(2):
            x = w*(.54+.12*i+.035*math.sin(seconds*.13+i))
            y = h*(.66+.08*i+.009*math.sin(seconds*.27+i))
            screen.blit(veil, (round(x), round(y)))
        for i in range(26):
            life = (seconds * (.022 + .002 * (i % 5)) + i * .617) % 1
            px = ((i * .38197 + .015 * math.sin(seconds * .23 + i) + life * .043) % 1) * w
            py = (.97 - .81 * life) * h
            pos = int(px), int(py)
            if .09 < life < .88:
                pygame.draw.circle(screen, (120, 113, 83) if i % 4 else (155, 112, 68),
                                   pos, max(1, round(scale)))
