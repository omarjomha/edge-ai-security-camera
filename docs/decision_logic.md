# Decision Logic: Tracks, Votes and the Intrusion State Machine

How the pipeline turns per-frame detections into a FRIEND / INTRUDER decision per person.
Code: `src/tracker.py` (tracking, face-to-track assignment, votes) and
`src/state_machine.py` (states, transitions, alarm). Settings in `config.yaml`.

## 1. Core rule

**YOLO triggers; face recognition only suppresses.** A person becomes INTRUDER through
**time alone** (no friend match within the grace period). A face can only move a person
toward FRIEND. Hidden, turned-away, masked or too-small faces therefore **fail safe**.

## 2. From frames to people

| Step | What it does | Why |
|---|---|---|
| Tracker (ByteTrack-style) | Gives each person a stable ID across frames | Evidence and timers are kept **per person**, not per frame |
| Face → track assignment | Each face is linked to **at most one** track (centre in the upper half of the person box, closest to the head position) | A friend's face can never vouch for a stranger standing next to them |
| Votes | Each track keeps its last `vote_window` usable face results (a friend's name, or "unknown") | One bad angle can't flip the decision |

Frames with no face, or a face smaller than `min_face_px`, add **no vote**: they are not
evidence either way.

## 3. States

```
new track ──► VERIFYING ──(friend votes ≥ vote_min_matches)────────► FRIEND   (suppress)
                  │  ▲                                                  │
                  │  └──(friend votes pushed out: re-verify, timer restarts)
                  └──(grace_period_s passes, no friend)──────────────► INTRUDER (ALERT)
                                                                        │
                  FRIEND ◄──(friend face finally matches: alarm cleared)┘
```

**Alarm** = at least one person currently in INTRUDER.

## 4. Votes vs. the grace timer

They answer different questions; the state machine needs both.

| | Votes (last 10 face results) | Grace timer (3 s) |
|---|---|---|
| Question | "Is this a friend?" | "How long have we been unable to tell?" |
| Advances | Only when a usable face is visible | Always |
| Can cause | VERIFYING → FRIEND | VERIFYING → INTRUDER |

Without the timer, a person who never shows a face would produce no votes and stay
VERIFYING forever: no alarm. The timer is the fail-safe deadline.

"Unknown" votes do **not** trigger INTRUDER early. That keeps a single path to the alarm
and gives a friend at a bad angle the full grace period. (Early alerts on strong unknown
evidence are possible, but not needed: the client accepts phone notifications within 30–60 s.)

## 5. Design decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **INTRUDER → FRIEND clears the alarm** when a friend's face matches later | A friend who walks in facing away triggers the alarm at 3 s, then cancels it by turning around. An intruder can't use this: it needs a real face match |
| 2 | **FRIEND → VERIFYING, timer restarts,** when unknown votes push the friend out of the window | Catches **ID switches** (section 6). Restarting the timer, instead of going straight to INTRUDER, avoids false alarms when the "unknown" votes come from the friend at a bad angle (G2) |
| 3 | **Alarm latching is not decided by the vision pipeline.** The pipeline reports events (`VERIFYING -> INTRUDER`, `left`, `alarm cleared`); the alarm currently turns off when the intruder's track is deleted (~2 s after leaving the view) | A real system usually keeps the alarm on until someone disarms it. Whether and how it latches belongs to **system integration** (alarm / app side, later phase), not the vision model |

## 6. Known risk: ID switches

An **ID switch** is when the tracker moves one person's ID onto another, typically when
people cross or overlap:

```
frame 100: you = #1 FRIEND, stranger = #2 INTRUDER
frame 101: paths cross, boxes overlap
frame 102: tracker matches #1 to the stranger  ->  stranger shown as "#1 FRIEND"
```

- **Covered:** if the stranger's face is visible afterwards, unknown votes push the friend
  out → VERIFYING → INTRUDER after the grace period (decision 2).
- **Not covered:** if the stranger keeps their face hidden after the switch, #1 gets **no
  votes** and stays FRIEND. Someone could exploit this by staying close to a friend, then
  walking off.
- Our tracker uses a simple alpha-beta motion model (not a Kalman filter), which makes
  switches somewhat more likely than in real ByteTrack.

Options (not implemented):

| Option | Effect | Trade-off |
|---|---|---|
| FRIEND expires after N s without a friend vote | Closes the gap | A friend facing away for long (TV, cooking) eventually triggers a false alarm (G2) |
| Re-verify both tracks after heavy overlap | Targets exactly the risky moment | Friend + guest walking together re-verify briefly; cheapest real fix |
| Better tracker (Kalman, appearance/ReID features) | Fewer switches overall | More complexity and compute |

**Plan:** keep decision 2 as is, record "two people crossing" clips in milestone 5 and
measure how often switches happen before choosing a fix.

## 7. Parameters

| Setting | Value | Meaning |
|---|---|---|
| `grace_period_s` | 3.0 | Time to prove "friend" before INTRUDER (was 5.0; shortened for faster alerts). Tune in milestone 5 (alert speed vs. friend false alarms) |
| `vote_window` | 10 | Face results remembered per track |
| `vote_min_matches` | 5 | Friend votes in the window to call FRIEND (~0.3 s at 15 fps) |
| `similarity_threshold` | 0.40 | Per-face match threshold (placeholder until the milestone 5 sweep) |
| `track_max_misses` | 30 | Frames a lost track (and its timer/state) is kept (~2 s at 15 fps) |

## 8. Verified scenarios (dev machine, 15 fps synthetic clips, grace period 3 s)

| Scenario | Result |
|---|---|
| Friend alone | FRIEND at 0.4 s, no alert |
| Stranger alone | INTRUDER at 3.1 s |
| Friend with face hidden throughout | INTRUDER at 3.9 s (fails safe; track started at 0.9 s) |
| Friend hidden 6 s, then face shown | INTRUDER at 3.9 s → FRIEND at 6.3 s (alarm cleared) |
| Friend + stranger together | Friend FRIEND at 0.4 s; stranger INTRUDER at 3.1 s |
| ID handed to a stranger (unit test) | FRIEND → VERIFYING at 1.5 s → INTRUDER at 4.5 s |
| Track deleted (unit test) | "left" event, alarm off |

These use still photos turned into video; real recorded clips are milestone 5.
