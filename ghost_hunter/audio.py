"""The crypt's ambience and event cues, mixed on independent SDL channels."""

from pathlib import Path

import pygame


class GameAudio:
    def __init__(self):
        self.muted = False
        self._quiet = False
        self._next_foot = 0
        self._channels = ()
        if not pygame.mixer.get_init():
            return
        directory = Path(__file__).with_name("assets") / "audio"
        self._channels = tuple(pygame.mixer.Channel(i) for i in range(5))
        self.ambience, self.boots, self.glass, self.outcome, self.interface = self._channels
        self._sounds = {
            name: pygame.mixer.Sound(str(directory / f"{name}.wav"))
            for name in ("crypt_air", "step_left", "step_right", "divining_glass", "capture", "escape",
                         "menu_start", "menu_select", "menu_back", "menu_exit", "wall")
        }
        self._mix()
        self.ambience.play(self._sounds["crypt_air"], loops=-1, fade_ms=800)

    @property
    def status(self):
        if not self._channels:
            return "Unavailable"
        return "Off" if self.muted else "On"

    def _mix(self):
        volumes = (.12 if self._quiet else .30, .33, .54, .90, .66)
        for channel, volume in zip(self._channels, volumes):
            channel.set_volume(0 if self.muted else volume)

    def toggle_mute(self):
        self.muted = not self.muted
        if self._channels:
            if self.muted:
                self.ambience.pause()
                for channel in self._channels[1:]:
                    channel.stop()
            else:
                self.ambience.unpause()
        self._mix()

    def quiet(self, enabled):
        if enabled != self._quiet:
            self._quiet = enabled
            self._mix()

    def begin_hunt(self):
        self.quiet(False)
        self._next_foot = 0
        if self._channels:
            self.outcome.fadeout(140)

    def footstep(self):
        if self._channels and not self.muted:
            name = ("step_left", "step_right")[self._next_foot]
            self.boots.play(self._sounds[name])
            self._next_foot = 1 - self._next_foot

    def sound_radar(self):
        if self._channels and not self.muted:
            self.glass.play(self._sounds["divining_glass"])

    def button(self, cue="select"):
        if self._channels and not self.muted:
            if cue == "exit":
                pygame.mixer.stop()
            self.interface.play(self._sounds[f"menu_{cue}"])

    def exit_cue_playing(self):
        return bool(self._channels and not self.muted and self.interface.get_busy())

    def wall_hit(self):
        if self._channels and not self.muted:
            self.boots.play(self._sounds["wall"])

    def finish(self, *, captured):
        self.quiet(True)
        if self._channels:
            self.boots.fadeout(45)
            self.glass.fadeout(65)
            if not self.muted:
                self.outcome.play(self._sounds["capture" if captured else "escape"])

    def close(self):
        self._channels = ()
        if pygame.mixer.get_init():
            pygame.mixer.stop()
            pygame.mixer.quit()
