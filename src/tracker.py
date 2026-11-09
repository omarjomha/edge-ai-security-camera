"""Person tracking, ByteTrack-style: stable IDs across frames.

Simplification vs. real ByteTrack: motion is predicted with an alpha-beta filter
(velocity nudged by each prediction error) instead of a Kalman filter.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

PERSON = 0                    # COCO class id
VELOCITY_GAIN = 0.3           # how strongly a prediction error corrects the velocity
LOST_VELOCITY_DECAY = 0.8     # unmatched tracks slow down instead of drifting away


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, 4) boxes vs (M, 4) boxes -> (N, M) IoU = overlap area / union area."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(tracks: list, boxes: np.ndarray, min_iou: float) -> tuple[list, list, list]:
    """Best one-to-one pairing (Hungarian algorithm) of tracks and boxes by IoU.

    Returns (pairs [(track_idx, box_idx)], unmatched track idxs, unmatched box idxs).
    """
    if not tracks or len(boxes) == 0:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    ious = iou_matrix(np.array([t.box for t in tracks]), boxes)
    rows, cols = linear_sum_assignment(-ious)          # maximise total IoU
    pairs = [(r, c) for r, c in zip(rows, cols) if ious[r, c] >= min_iou]
    matched_t = {r for r, _ in pairs}
    matched_b = {c for _, c in pairs}
    return (pairs, [i for i in range(len(tracks)) if i not in matched_t],
            [j for j in range(len(boxes)) if j not in matched_b])


class Track:
    def __init__(self, box: np.ndarray, score: float):
        self.box = box.astype(np.float32)        # x1, y1, x2, y2 (frame pixels)
        self.velocity = np.zeros(4, np.float32)  # box change per frame
        self.score = score
        self.track_id = None                     # assigned once confirmed
        self.hits = 1                            # frames matched so far
        self.misses = 0                          # consecutive frames without a match

    def predict(self) -> None:
        self.box = self.box + self.velocity      # assume the person keeps moving the same way

    def update(self, box: np.ndarray, score: float) -> None:
        error = box - self.box                   # how far off the prediction was
        self.velocity += VELOCITY_GAIN * error
        self.box, self.score = box.astype(np.float32), score
        self.hits += 1
        self.misses = 0

    def mark_missed(self) -> None:
        self.misses += 1
        self.velocity *= LOST_VELOCITY_DECAY


class Tracker:
    def __init__(self, high_conf: float, new_conf: float, match_iou: float,
                 max_misses: int, min_hits: int):
        self.high_conf = high_conf    # boxes >= this are "confident" (stage 1)
        self.new_conf = new_conf      # only boxes >= this may start a new track
        self.match_iou = match_iou    # minimum overlap to accept a stage-1 match
        self.max_misses = max_misses  # frames a lost track is kept before deletion
        self.min_hits = min_hits      # matched frames before a track gets an ID
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, dets: np.ndarray) -> list[Track]:
        """(N, 6) detections for one frame -> confirmed tracks seen in this frame."""
        people = dets[dets[:, 5] == PERSON]
        high = people[people[:, 4] >= self.high_conf]
        low = people[people[:, 4] < self.high_conf]

        for t in self.tracks:
            t.predict()

        # Stage 1: every track (including recently lost ones) vs. confident boxes
        pairs, left_t, left_h = match(self.tracks, high[:, :4], self.match_iou)
        for ti, bi in pairs:
            self.tracks[ti].update(high[bi, :4], high[bi, 4])

        # Stage 2 (the ByteTrack idea): tracks that were visible last frame but got no
        # confident box try the weak boxes (e.g. a half-hidden person). Strict IoU here.
        candidates = [self.tracks[i] for i in left_t if self.tracks[i].misses == 0]
        pairs2, _, _ = match(candidates, low[:, :4], min_iou=0.5)
        rescued = set()
        for ci, bi in pairs2:
            candidates[ci].update(low[bi, :4], low[bi, 4])
            rescued.add(id(candidates[ci]))

        for i in left_t:
            if id(self.tracks[i]) not in rescued:
                self.tracks[i].mark_missed()

        # New tracks only from unmatched, clearly confident boxes
        for bi in left_h:
            if high[bi, 4] >= self.new_conf:
                self.tracks.append(Track(high[bi, :4], high[bi, 4]))

        # Delete tracks lost for too long, and unconfirmed ones that already missed a
        # frame (likely a one-off false detection); give an ID to newly confirmed ones
        self.tracks = [t for t in self.tracks
                       if t.misses <= self.max_misses and not (t.track_id is None and t.misses > 0)]
        for t in self.tracks:
            if t.track_id is None and t.hits >= self.min_hits:
                t.track_id = self.next_id
                self.next_id += 1

        return [t for t in self.tracks if t.track_id is not None and t.misses == 0]
