"""Per-person intrusion state machine.

    new track -> VERIFYING -> FRIEND    (friend votes reached: suppress)
                     |
                     +-------> INTRUDER  (grace period over, no friend: ALERT)

Core rule: reaching INTRUDER needs no face, only time (back turned, face hidden or
too small all fail safe). Face recognition can only move a person toward FRIEND.
"""
from dataclasses import dataclass

VERIFYING, FRIEND, INTRUDER = "VERIFYING", "FRIEND", "INTRUDER"


@dataclass
class PersonState:
    state: str
    since: float                 # when the current state started (seconds)
    friend: str | None = None    # who, if FRIEND


class StateMachine:
    def __init__(self, grace_period_s: float, min_matches: int):
        self.grace_period_s = grace_period_s
        self.min_matches = min_matches            # friend votes needed (see Track.identity)
        self.people: dict[int, PersonState] = {}  # track_id -> state

    def update(self, tracks: list, now: float) -> list[str]:
        """Advance every confirmed track (visible or briefly lost). Returns event messages."""
        events = []
        alive = set()
        for t in tracks:
            if t.track_id is None:                 # not confirmed yet: no state
                continue
            alive.add(t.track_id)
            p = self.people.get(t.track_id)
            if p is None:
                p = self.people[t.track_id] = PersonState(VERIFYING, now)
                events.append(f"#{t.track_id} new person -> VERIFYING")

            name, votes = t.identity(self.min_matches)
            if name is not None and p.state != FRIEND:
                note = " (alarm cleared)" if p.state == INTRUDER else ""
                events.append(f"#{t.track_id} {p.state} -> FRIEND {name} ({votes} votes){note}")
                p.state, p.since, p.friend = FRIEND, now, name
            elif name is None and p.state == FRIEND:
                # Unknown votes pushed the friend out (e.g. tracker handed the ID to
                # someone else): verify again from scratch, with a fresh grace period
                events.append(f"#{t.track_id} FRIEND -> VERIFYING (friend match lost)")
                p.state, p.since, p.friend = VERIFYING, now, None
            elif p.state == VERIFYING and now - p.since >= self.grace_period_s:
                events.append(f"#{t.track_id} VERIFYING -> INTRUDER: no friend match within "
                              f"{self.grace_period_s:.1f} s  *** ALERT ***")
                p.state, p.since = INTRUDER, now

        for track_id in list(self.people):        # tracker deleted the track: person left
            if track_id not in alive:
                events.append(f"#{track_id} left (was {self.people[track_id].state})")
                del self.people[track_id]
        return events

    @property
    def alarm(self) -> bool:
        return any(p.state == INTRUDER for p in self.people.values())
