"""Per-person intrusion state machine.

    new track -> VERIFYING -> FRIEND    (friend votes reached: suppress)
                     |
                     +-------> GUEST     (grace period over, no match, but a verified
                     |                    FRIEND who appeared with them is in view: no alarm)
                     +-------> INTRUDER  (grace period over, no friend: ALERT)
                                  |
                                  +----> GUEST (alarm cleared) if a friend who appeared
                                         with them is verified within guest_window_s

"Appeared with them" = both tracks started within guest_arrival_gap_s of each other, so a
friend already in the room (or arriving later, e.g. during a break-in) never vouches.

Core rule: reaching INTRUDER needs no face, only time (back turned, face hidden or
too small all fail safe). Face recognition can only move a person toward FRIEND, and a
visible FRIEND vouches for companions who arrived with them (GUEST) if allow_guests is on.
Trade-off: an intruder slipping in with/behind a friend (tailgating) also becomes GUEST.
"""
from dataclasses import dataclass

VERIFYING, FRIEND, GUEST, INTRUDER = "VERIFYING", "FRIEND", "GUEST", "INTRUDER"


@dataclass
class PersonState:
    state: str
    since: float                 # when the current state started (seconds)
    first_seen: float            # when this person appeared (for the guest window)
    friend: str | None = None    # who, if FRIEND
    host: str | None = None      # which friend vouched for them, if GUEST


class StateMachine:
    def __init__(self, grace_period_s: float, min_matches: int, allow_guests: bool = True,
                 guest_window_s: float = 10.0, guest_arrival_gap_s: float = 2.0):
        self.grace_period_s = grace_period_s
        self.min_matches = min_matches            # friend votes needed (see Track.identity)
        self.allow_guests = allow_guests          # visible FRIEND turns companions into GUEST
        self.guest_window_s = guest_window_s      # friend verified this soon after arrival
        self.guest_arrival_gap_s = guest_arrival_gap_s  # max gap between their arrivals
        self.people: dict[int, PersonState] = {}  # track_id -> state

    def update(self, tracks: list, now: float) -> list[str]:
        """Advance every confirmed track (visible or briefly lost). Returns event messages."""
        events = []
        confirmed = [t for t in tracks if t.track_id is not None]  # unconfirmed: no state yet

        # Pass 1: new people, and friend matches gained or lost (from face votes)
        for t in confirmed:
            p = self.people.get(t.track_id)
            if p is None:
                p = self.people[t.track_id] = PersonState(VERIFYING, now, first_seen=now)
                events.append(f"#{t.track_id} new person -> VERIFYING")

            name, votes = t.identity(self.min_matches)
            if name is not None and p.state != FRIEND:
                note = " (alarm cleared)" if p.state == INTRUDER else ""
                events.append(f"#{t.track_id} {p.state} -> FRIEND {name} ({votes} votes){note}")
                p.state, p.since, p.friend, p.host = FRIEND, now, name, None
            elif name is None and p.state == FRIEND:
                # Unknown votes pushed the friend out (e.g. tracker handed the ID to
                # someone else): verify again from scratch, with a fresh grace period
                events.append(f"#{t.track_id} FRIEND -> VERIFYING (friend match lost)")
                p.state, p.since, p.friend = VERIFYING, now, None

        # Verified friends visible in this frame (misses == 0)
        hosts = [(t.track_id, self.people[t.track_id]) for t in confirmed
                 if t.misses == 0 and self.people[t.track_id].state == FRIEND]

        def companions(p: PersonState) -> list:
            """Visible friends who appeared within guest_arrival_gap_s of this person."""
            return [(hid, h) for hid, h in hosts
                    if abs(h.first_seen - p.first_seen) <= self.guest_arrival_gap_s]

        for t in confirmed:
            p = self.people[t.track_id]
            # Pass 2: grace period over without a friend match -> GUEST or INTRUDER
            if p.state == VERIFYING and now - p.since >= self.grace_period_s:
                with_them = companions(p) if self.allow_guests else []
                if with_them:
                    host_id, host_name = with_them[0][0], with_them[0][1].friend
                    events.append(f"#{t.track_id} VERIFYING -> GUEST: no friend match, but "
                                  f"accompanied by FRIEND {host_name} (#{host_id})")
                    p.state, p.since, p.host = GUEST, now, host_name
                else:
                    events.append(f"#{t.track_id} VERIFYING -> INTRUDER: no friend match within "
                                  f"{self.grace_period_s:.1f} s  *** ALERT ***")
                    p.state, p.since = INTRUDER, now
            # Pass 3: the friend they arrived with was hidden or back-turned at the 3 s
            # mark but got verified within guest_window_s of this person arriving.
            elif p.state == INTRUDER and self.allow_guests:
                together = [(hid, h) for hid, h in companions(p)
                            if h.since <= p.first_seen + self.guest_window_s]
                if together:
                    host_id, host_name = together[0][0], together[0][1].friend
                    events.append(f"#{t.track_id} INTRUDER -> GUEST: arrived with FRIEND "
                                  f"{host_name} (#{host_id}) (alarm cleared)")
                    p.state, p.since, p.host = GUEST, now, host_name

        alive = {t.track_id for t in confirmed}
        for track_id in list(self.people):        # tracker deleted the track: person left
            if track_id not in alive:
                events.append(f"#{track_id} left (was {self.people[track_id].state})")
                del self.people[track_id]
        return events

    @property
    def alarm(self) -> bool:
        return any(p.state == INTRUDER for p in self.people.values())
