"""Interactive pages of the crypt, shared by the solo and competitive hunts."""

import sqlite3

import pygame

from config import DIFFICULTY_CONFIGS
from ghost_hunter.crypt_theme import BRASS, INK, IVORY, PARCHMENT, ROSE, RULE, VERDIGRIS, plaque, sigil
from ghost_hunter.leaderboard import MAX_NAME_LENGTH, MODES


class CloseGame(Exception):
    """A window-close event from a nested menu."""


def window_closing(events):
    return any(event.type in (pygame.QUIT, pygame.WINDOWCLOSE) for event in events)


def _events(gui, fullscreen=True):
    events = pygame.event.get()
    # SDL removes the whole batch from the queue. A button must never swallow a later close.
    if window_closing(events):
        raise CloseGame
    for event in events:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F2:
            gui.audio.toggle_mute()
            continue
        if fullscreen and event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            gui.toggle_fullscreen()
            break
        yield event


def _present(gui):
    pygame.display.flip()
    gui.clock.tick(30)


def _pressed(event, *keys):
    return event.type == pygame.KEYDOWN and event.key in keys


def _navigation(event, selected, count):
    if _pressed(event, pygame.K_UP):
        return (selected - 1) % count
    if _pressed(event, pygame.K_DOWN):
        return (selected + 1) % count
    return selected


def _clicked(event, rect):
    return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and rect.collidepoint(event.pos)


def _activated(gui, event, rect, *keys, cue="select"):
    if _clicked(event, rect) or _pressed(event, *keys):
        gui.audio.button(cue)
        return True
    return False


def _start_choice(gui, choice):
    cue = {"start": "start", "exit": "exit"}.get(choice, "select")
    gui.audio.button(cue)
    if choice == "exit":
        # The door sounds while the window is still open. OS-close always wins immediately.
        deadline = pygame.time.get_ticks() + 250
        while gui.audio.exit_cue_playing() and pygame.time.get_ticks() < deadline:
            for _ in _events(gui, fullscreen=False):
                pass
            gui.clock.tick(120)
    return choice


def _button(gui, rect, label, accent=BRASS, **style):
    gui._menu_button(rect, label, rect.collidepoint(pygame.mouse.get_pos()), accent, **style)


def _line(gui, font, text, color, rect, *, right=False):
    ink = gui._text(font, text, color, rect.width, rect.height)
    where = ink.get_rect(midright=rect.midright) if right else ink.get_rect(midleft=rect.midleft)
    gui.screen.blit(ink, where)


def _back(gui, pane):
    height = min(pane.height // 10, round(55 * gui.ui_scale))
    rect = pygame.Rect(pane.left, pane.bottom-height, min(pane.width//2, round(218 * gui.ui_scale)), height)
    _button(gui, rect, "Return", hint="Esc")
    return rect


def start_screen(gui):
    items = (("Enter the Crypt", "lantern", "Enter", "start"),
             ("Instructions", "book", "I", "instructions"),
             ("Leaderboard", "hourglass", "L", "leaderboard"),
             ("Exit", "door", "Esc", "exit"))
    selected = 0
    while True:
        pane = gui._menu_page("Ghost Hunter", "Tracking an Invisible Enemy", hero=True)
        s = gui.ui_scale
        # The blood traces the very top of the frame, away from the lettering.
        margin = round(18*s)
        gui._draw_blood(pygame.Rect(margin+round(44*s), margin, min(pane.width//2, round(260*s)), 1))
        gui._draw_cobweb(gui.screen.get_width()-margin, margin, max(12, round(70*s)), "TR")
        height = min(round(78*s), pane.height//6)
        gap = max(3, round(10*s))
        buttons = []
        for i, (label, motif, hint, _) in enumerate(items):
            rect = pygame.Rect(pane.left, pane.top + round(12*s) + i*(height+gap),
                               min(pane.width, round(548*s)), height)
            _button(gui, rect, label, motif=motif, hint=hint if i or selected == 0 else None, primary=i==selected)
            buttons.append(rect)
        _line(gui, gui.font_mono, f"Up / Down   Select     Enter   Open     F   Fullscreen     F2   Sound {gui.audio.status}", PARCHMENT,
              pygame.Rect(pane.left, pane.bottom-3*gap, pane.width, 3*gap))
        _present(gui)
        for event in _events(gui):
            selected = _navigation(event, selected, len(buttons))
            if _pressed(event, pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                return _start_choice(gui, items[selected][3])
            for i, rect in enumerate(buttons):
                keys = ((), (pygame.K_i,), (pygame.K_l,), (pygame.K_ESCAPE,))[i]
                if _clicked(event, rect) or _pressed(event, *keys):
                    return _start_choice(gui, items[i][3])


def choose_mode(gui):
    details = ("Track the phantom by its noisy signal.",
               "Catch the ghost. A faster time and fewer moves earn a higher score.",
               "Share one keyboard. The first hunter to reach the ghost wins.")
    selected = 0
    while True:
        pane = gui._menu_page("Choose Your Mode")
        back = _back(gui, pane)
        gap = max(4, round(13*gui.ui_scale))
        height = min((back.top - pane.top) // len(MODES) - gap, round(114*gui.ui_scale))
        buttons = []
        for i, (mode, detail, motif) in enumerate(zip(MODES, details, ("lantern", "eye", "duel"))):
            rect = pygame.Rect(pane.left, pane.top + i*(height+gap), pane.width, height)
            _button(gui, rect, mode, (BRASS, VERDIGRIS, ROSE)[i], motif=motif, hint=str(i+1),
                    detail=detail, primary=i==selected)
            buttons.append(rect)
        _present(gui)
        for event in _events(gui):
            selected = _navigation(event, selected, len(buttons))
            if _pressed(event, pygame.K_RETURN, pygame.K_KP_ENTER):
                gui.audio.button()
                return MODES[selected]
            if _activated(gui, event, back, pygame.K_ESCAPE, cue="back"):
                return None
            for i, rect in enumerate(buttons):
                if _activated(gui, event, rect, pygame.K_1+i):
                    return MODES[i]


def choose_difficulty(gui, *, demo=False):
    selected = 0
    while True:
        pane = gui._menu_page("Choose Your Fate", "Select the difficulty of the hunt.")
        height = min(pane.height//6, round(94*gui.ui_scale))
        gap = max(3, round(10*gui.ui_scale))
        buttons = []
        for i, (name, cfg) in enumerate(DIFFICULTY_CONFIGS.items()):
            rect = pygame.Rect(pane.left, pane.top+i*(height+gap), pane.width, height)
            timing = "One ghost move per turn" if demo else f"Ghost moves every {cfg['ghost_speed']:g}s"
            detail = f"{cfg['width']} × {cfg['height']} cells    ·    {cfg['turn_limit']} turns    ·    {timing}"
            accent = (VERDIGRIS, BRASS, (237, 167, 124), ROSE)[i]
            _button(gui, rect, name, accent, motif=("lantern", "eye", "hourglass", "door")[i],
                    hint=str(i+1), detail=detail, primary=i==selected)
            buttons.append((rect, name))
        back = _back(gui, pane)
        _present(gui)
        for event in _events(gui):
            selected = _navigation(event, selected, len(buttons))
            if _pressed(event, pygame.K_RETURN, pygame.K_KP_ENTER):
                gui.audio.button()
                return buttons[selected][1]
            if _activated(gui, event, back, pygame.K_ESCAPE, cue="back"):
                return None
            for i, (rect, name) in enumerate(buttons):
                if _activated(gui, event, rect, pygame.K_1+i):
                    return name


def instructions_screen(gui):
    chapters = (
        ("lantern", "The hunt", ("Catch the moving ghost before your turns run out.",
                                "Move with WASD or the arrows. In a duel, each hunter uses one set.")),
        ("eye", "Read the belief map", ("Each noisy distance ping updates the HMM. The reticle marks its best estimate.",
                                       "H switches between proximity rings and probability. Gold marks peak belief.")),
        ("hourglass", "Field controls", ("X lights a path. G reveals the ghost in Solo. P pauses. F toggles fullscreen. F2 toggles sound.",
                                       "The AI demo follows the most likely cell using only the noisy readings.")),
    )
    while True:
        pane = gui._menu_page("Instructions", "A field guide to the hunt", wide=True)
        height = min(pane.height//5, round(125*gui.ui_scale))
        gap = max(3, round(12*gui.ui_scale))
        for i, (motif, heading, lines) in enumerate(chapters):
            rect = pygame.Rect(pane.left, pane.top+i*(height+gap), pane.width, height)
            plaque(gui.screen, rect, INK, RULE, gui.ui_scale)
            pad = max(4, round(16*gui.ui_scale))
            radius = min(height//8, round(14*gui.ui_scale))
            sigil(gui.screen, (rect.left+pad+radius, rect.top+height//4), radius, motif, BRASS)
            heading_box = pygame.Rect(rect.left+2*(pad+radius), rect.top+2,
                                      rect.width-3*pad-2*radius, height*42//100)
            _line(gui, gui.font_title, heading, IVORY, heading_box)
            for j, line in enumerate(lines):
                box = pygame.Rect(rect.left+pad, rect.top+round((.44+.25*j)*height),
                                  rect.width-2*pad, height//4)
                _line(gui, gui.font_body, line, PARCHMENT, box)
        back = _back(gui, pane)
        demo = pygame.Rect(pane.left, back.top-gap-back.height, (pane.width-gap)//2, back.height)
        entropy = demo.move(demo.width+gap, 0)
        _button(gui, demo, "Watch AI Hunt", VERDIGRIS, motif="eye", hint="D", primary=True)
        _button(gui, entropy, "Watch Entropy Hunter", BRASS, motif="hourglass", hint="E")
        _present(gui)
        for event in _events(gui):
            if _activated(gui, event, demo, pygame.K_d, cue="start"):
                return "demo"
            if _activated(gui, event, entropy, pygame.K_e, cue="start"):
                return "entropy"
            if _activated(gui, event, back, pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_KP_ENTER, cue="back"):
                return None


def enter_names(gui, mode, existing=()):
    count = 2 if mode == "Human vs Human" else 1
    names = (list(existing) + [""] * count)[:count]
    selected = 0
    pygame.key.start_text_input()
    try:
        while True:
            pane = gui._menu_page("Name Your Hunters" if count == 2 else "Name Your Hunter", mode)
            height = min(pane.height//6, round(86*gui.ui_scale))
            gap = max(4, round(12*gui.ui_scale))
            fields = []
            for i, name in enumerate(names):
                y = pane.top+i*(height+4*gap)
                label = ("Player 1  ·  WASD", "Player 2  ·  Arrows")[i] if count == 2 else "Hunter name"
                _line(gui, gui.font_body, label, PARCHMENT, pygame.Rect(pane.left, y, pane.width, 2*gap))
                field = pygame.Rect(pane.left, y+3*gap, pane.width, height)
                accent = BRASS if i == selected else RULE
                plaque(gui.screen, field, INK, accent, gui.ui_scale)
                content = field.inflate(-4*gap, -gap)
                value = gui._text(gui.font_body, name, IVORY, content.width-gap, content.height)
                gui.screen.blit(value, value.get_rect(midleft=content.midleft))
                if i == selected and pygame.time.get_ticks() % 1000 < 600:
                    x = content.left+value.get_width()+2
                    pygame.draw.line(gui.screen, accent, (x, field.centery-height//5),
                                     (x, field.centery+height//5), max(1, round(gui.ui_scale)))
                fields.append(field)
            pygame.key.set_text_input_rect(fields[selected])
            ready = all(name.strip() for name in names)
            start = pygame.Rect(pane.left, fields[-1].bottom+2*gap,
                                min(pane.width, round(410*gui.ui_scale)), height)
            _button(gui, start, "Begin Hunt" if ready else "Enter a name" if count==1 else "Enter both names",
                    VERDIGRIS if ready else PARCHMENT, motif="lantern", hint="Enter", primary=ready)
            back = _back(gui, pane)
            _present(gui)
            for event in _events(gui, fullscreen=False):
                ready = all(name.strip() for name in names)
                if _activated(gui, event, back, pygame.K_ESCAPE, cue="back"):
                    return None
                if ready and _activated(gui, event, start, pygame.K_RETURN, pygame.K_KP_ENTER, cue="start"):
                    return tuple(name.strip() for name in names)
                if event.type == pygame.TEXTINPUT:
                    typed = "".join(char for char in event.text if char.isprintable())
                    names[selected] = (names[selected]+typed)[:MAX_NAME_LENGTH]
                elif _pressed(event, pygame.K_BACKSPACE):
                    names[selected] = names[selected][:-1]
                elif _pressed(event, pygame.K_TAB):
                    selected = (selected+1) % count
                elif _pressed(event, pygame.K_RETURN, pygame.K_KP_ENTER):
                    selected = next(i for i, name in enumerate(names) if not name.strip())
                for i, field in enumerate(fields):
                    if _clicked(event, field):
                        selected = i
    finally:
        pygame.key.stop_text_input()


def show_leaderboard(gui, store):
    mode_index = difficulty_index = 0
    difficulties = tuple(DIFFICULTY_CONFIGS)
    loaded, rows, message = None, [], ""
    while True:
        pane = gui._menu_page("Capture Leaderboard", "Ranked by active hunt time. Pauses are excluded.", wide=True)
        mode, difficulty = MODES[mode_index], difficulties[difficulty_index]
        if loaded != (mode, difficulty):
            try:
                rows = store.top(mode, difficulty)
                message = "" if rows else "No captures recorded for this mode and difficulty."
            except (sqlite3.Error, OSError):
                rows, message = [], "The local leaderboard could not be read."
            loaded = (mode, difficulty)
        gap = max(4, round(14*gui.ui_scale))
        back = _back(gui, pane)
        fh = min(pane.height//10, round(67*gui.ui_scale))
        filters = (pygame.Rect(pane.left, pane.top, (pane.width-gap)//2, fh),
                   pygame.Rect(pane.left+(pane.width+gap)//2, pane.top, (pane.width-gap)//2, fh))
        _button(gui, filters[0], mode, VERDIGRIS, hint="M")
        _button(gui, filters[1], difficulty, BRASS, hint="D")
        table = pygame.Rect(pane.left, pane.top+fh+gap, pane.width, back.top-pane.top-fh-2*gap)
        rh = max(1, min(round(48*gui.ui_scale), (table.height-2*gap)//11))
        table.height = min(table.height, 2*gap + rh*max(4, len(rows)+1))
        plaque(gui.screen, table, INK, RULE, gui.ui_scale)
        inner = table.inflate(-2*gap, -2*gap)
        # Cells share column boundaries even for long player names and four-digit times.
        edges = [inner.left+round(inner.width*x) for x in (0, .105, .55, .75, 1)]
        for i, row in enumerate([("Rank", "Hunter", "Time", "Date (UTC)")]+[
                (f"{rank:02d}", name, f"{ms/1000:.3f}s", stamp[:10])
                for rank, (name, ms, stamp) in enumerate(rows, 1)]):
            top = inner.top+i*rh
            if i == 1:
                pygame.draw.rect(gui.screen, (45, 40, 29), (inner.x, top, inner.width, rh))
            for j, value in enumerate(row):
                box = pygame.Rect(edges[j]+gap//2, top, edges[j+1]-edges[j]-gap, rh)
                color = VERDIGRIS if i==0 else BRASS if i==1 else IVORY
                _line(gui, gui.font_body, value, color, box, right=j>=2)
            pygame.draw.line(gui.screen, RULE if i==0 else (53, 53, 44),
                             (inner.left, top+rh-1), (inner.right, top+rh-1))
        if message:
            _line(gui, gui.font_body, message, PARCHMENT, pygame.Rect(inner.left, inner.top+rh, inner.width, 2*rh))
        _present(gui)
        for event in _events(gui):
            if _activated(gui, event, back, pygame.K_ESCAPE, cue="back"):
                return
            if _activated(gui, event, filters[0], pygame.K_m):
                mode_index = (mode_index+1) % len(MODES)
            if _activated(gui, event, filters[1], pygame.K_d):
                difficulty_index = (difficulty_index+1) % len(difficulties)


def _draw_scorecard(gui, pane, card, player_name, step_limit):
    height = min(round(373 * gui.ui_scale), pane.height * 3 // 5)
    panel = pygame.Rect(pane.topleft, (pane.width, height))
    winners = card.winners
    accent = VERDIGRIS if winners == (0,) else BRASS
    plaque(gui.screen, panel, INK, accent, gui.ui_scale)
    pad = max(5, round(20 * gui.ui_scale))
    inner = panel.inflate(-2 * pad, -pad)
    header_h = inner.height * 17 // 100
    header = pygame.Rect(inner.left, inner.top, inner.width, header_h)
    radius = max(3, min(pad, header_h // 3))
    sigil(gui.screen, (header.left + radius, header.centery), radius, "lantern", accent)
    verdict_box = pygame.Rect(header.left + 3 * radius, header.top, header.width // 2 - 3 * radius, header_h)
    _line(gui, gui.font_title, card.verdict, IVORY, verdict_box)
    if step_limit is not None:
        detail = f"{step_limit}-turn limit reached"
    else:
        detail = "Both hunters captured the ghost"
    _line(gui, gui.font_body, detail, PARCHMENT,
          pygame.Rect(header.centerx, header.y, header.width // 2, header_h), right=True)
    pygame.draw.line(gui.screen, RULE, header.bottomleft, (header.right, header.bottom))

    edges = [inner.left + round(inner.width * fraction) for fraction in (0, .41, .56, .80, 1)]
    def cell(column, y, h):
        return pygame.Rect(edges[column] + pad // 2, y, edges[column + 1] - edges[column] - pad, h)

    label_h = inner.height * 12 // 100
    for j, label in enumerate(("Hunter", "Moves", "Active time", "Total score")):
        _line(gui, gui.font_mono, label, PARCHMENT, cell(j, header.bottom + 1, label_h), right=j > 0)
    y = header.bottom + 1 + label_h
    row_h = inner.height * 23 // 100
    for i, (hunter, name) in enumerate(zip(card.hunters, (player_name, "AI"))):
        row = pygame.Rect(inner.left, y, inner.width, row_h)
        if i in winners:
            pygame.draw.rect(gui.screen, (30, 41, 35) if i == 0 else (44, 38, 28), row)
        role = "You" if i == 0 else "Rival"
        status = "Captured" if hunter.captured else "Did not finish"
        _line(gui, gui.font_body, name, IVORY, cell(0, y, row_h * 55 // 100))
        _line(gui, gui.font_mono, f"{role}  /  {status}", VERDIGRIS if i == 0 else BRASS,
              cell(0, y + row_h * 55 // 100, row_h * 36 // 100))
        for j, value in enumerate((str(hunter.moves), f"{hunter.time_ms / 1000:.3f} s", str(hunter.points)), 1):
            _line(gui, gui.font_badge if j == 3 else gui.font_mono, value,
                  accent if j == 3 and i in winners else IVORY, cell(j, y, row_h), right=True)
        y += row_h
        pygame.draw.line(gui.screen, RULE, (inner.left, y), (inner.right, y))
    foot_h = (inner.bottom - y) // 2
    _line(gui, gui.font_body, "Capture score = 1,000 − (10 × active seconds) − moves", IVORY,
          pygame.Rect(inner.left, y + 1, inner.width, foot_h))
    _line(gui, gui.font_mono, "Time rounds to 0.1 s. A capture earns at least 1 point. An unfinished hunt scores 0.",
          PARCHMENT, pygame.Rect(inner.left, y + foot_h, inner.width, foot_h))
    return panel


def _draw_capture_summary(gui, pane, winners, elapsed, step_limit, accent, motif):
    captured = step_limit is None
    if captured:
        result = f"{winners[0]} wins" if len(winners) == 1 else "Tied hunters · " + " & ".join(winners)
        detail = f"Capture time   {elapsed:.3f}s"
    else:
        result = "The ghost escaped."
        detail = f"All {step_limit} steps were used before capture."
    height = min(round(142*gui.ui_scale), pane.height//4)
    panel = pygame.Rect(pane.left, pane.top, pane.width, height)
    plaque(gui.screen, panel, INK, accent, gui.ui_scale)
    r = max(4, height//5)
    if captured:
        cage = pygame.Rect(0, 0, 3*r, round(height*.79))
        cage.center = (panel.left+2*r, panel.centery)
        gui.actors.reliquary(gui.screen, cage, 1.0)
    else:
        sigil(gui.screen, (panel.left+2*r, panel.centery), r, motif, accent)
    tx = panel.left+4*r
    _line(gui, gui.font_title, result, IVORY,
          pygame.Rect(tx, panel.top+height//9, panel.right-tx-r, height//2))
    _line(gui, gui.font_body, detail, PARCHMENT,
          pygame.Rect(tx, panel.centery, panel.right-tx-r, height//3))
    return panel


def show_capture_result(gui, winners, elapsed, save_error=None, *, draw_capture=None, step_limit=None,
                        scorecard=None, player_name=None, difficulty=None):
    captured = bool(scorecard.winners) if scorecard is not None else step_limit is None
    try:
        if draw_capture is not None:
            until = pygame.time.get_ticks() + 1800
            while pygame.time.get_ticks() < until:
                draw_capture()
                for event in _events(gui):
                    if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                        until = 0
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        until = 0
                gui.clock.tick(30)
        while True:
            accent, motif = (BRASS, "lantern") if captured else (ROSE, "hourglass")
            if scorecard is not None:
                subtitle = f"AI vs Human  /  {difficulty}" if difficulty else "AI vs Human"
                pane = gui._menu_page("Race Scorecard", subtitle, wide=True, mood="capture" if captured else "escape")
                panel = _draw_scorecard(gui, pane, scorecard, player_name, step_limit)
                button_height = min(pane.height // 9, round(67 * gui.ui_scale))
            else:
                title = "Ghost Captured" if captured else "Steps Exhausted"
                pane = gui._menu_page(title, mood="capture" if captured else "escape")
                panel = _draw_capture_summary(gui, pane, winners, elapsed, step_limit, accent, motif)
                button_height = panel.height // 2
            gap = max(4, round(13*gui.ui_scale))
            y = panel.bottom+2*gap
            if save_error:
                _line(gui, gui.font_body, save_error, ROSE, pygame.Rect(pane.left, y, pane.width, 3*gap))
                y += 4*gap
            replay = pygame.Rect(pane.left, y, min(pane.width, round(470*gui.ui_scale)), max(1, button_height))
            leave = replay.move(0, replay.height+gap)
            _button(gui, replay, "Play Again" if captured else "Try Again", accent, motif="lantern", hint="Enter", primary=True)
            _button(gui, leave, "Exit to Main Menu", motif="door", hint="Esc")
            _present(gui)
            for event in _events(gui):
                if _activated(gui, event, replay, pygame.K_RETURN, pygame.K_KP_ENTER, cue="start"):
                    return "restart"
                if _activated(gui, event, leave, pygame.K_ESCAPE, cue="back"):
                    return "menu"
    except CloseGame:
        return "exit"
