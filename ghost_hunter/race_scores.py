"""Individual finish times and the scoring rule for AI vs Human."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HunterScore:
    moves: int
    time_ms: int
    captured: bool

    @property
    def points(self) -> int:
        if not self.captured:
            return 0
        tenths = (self.time_ms + 50) // 100
        return max(1, 1000 - tenths - self.moves)


@dataclass(frozen=True)
class RaceScorecard:
    hunters: tuple[HunterScore, HunterScore]

    @property
    def winners(self) -> tuple[int, ...]:
        best = max(hunter.points for hunter in self.hunters)
        return tuple(i for i, hunter in enumerate(self.hunters) if best and hunter.points == best)

    @property
    def verdict(self) -> str:
        return {(0,): "You win", (1,): "AI wins", (0, 1): "Draw", (): "Draw"}[self.winners]
