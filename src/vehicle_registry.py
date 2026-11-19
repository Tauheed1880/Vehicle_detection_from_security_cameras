"""
Step 9 - New vehicle?

The tracker reports a track ID on EVERY frame. This class:

  * remembers which IDs have already been analysed, so the heavy models run
    only once per vehicle;
  * remembers the best-quality crop seen for every ID, so a vehicle that never
    passed the box filter can still be analysed when its track disappears,
    instead of being dropped silently;
  * watches each new track for a few frames (box stability, sharpness, whether
    the crop is still improving) so the best crop - not the first one - is sent.
"""
from src.cropper import box_iou


class VehicleRegistry:
    def __init__(self, track_timeout=30):
        self._done = set()          # analysed / handed to the worker
        self._seen = set()          # every track ID ever reported
        self._first_seen = {}       # id -> frame number it was first detected on
        self._first_when = {}       # id -> local datetime of that first detection
        self._first_det = {}        # id -> detection dict from its first frame
        self._best = {}             # id -> (good, score, area, det, crop)
        self._cand = {}             # id -> [(score, crop)] best-first, capped
        self._last_seen = {}        # id -> last frame number it was reported
        self._prev_box = {}         # id -> previous frame's box (stability check)
        self._stable_run = {}       # id -> consecutive stable frames
        self._last_score = {}       # id -> most recent crop quality score
        self._last_sharp = {}       # id -> most recent crop sharpness
        self._observed = {}         # id -> frames this track has been seen
        self.track_timeout = track_timeout

    # ------------------------------------------------ step 9: one analysis
    def is_new(self, track_id):
        """True if this vehicle has not been analysed yet."""
        return track_id not in self._done

    def mark_first_seen(self, track_id, frame_no, det=None, when=None):
        """Remember this vehicle's FIRST detection (kept forever).

        Stores the frame number, the detection dict of that first frame and the
        local wall-clock instant it was captured. All three use setdefault, so
        they never move once recorded - every record for this track (normal or
        fallback) then shows the first-tracked time, not a later frame's.
        `det`/`when` are optional so callers that only know the frame number
        still work.
        """
        self._first_seen.setdefault(track_id, frame_no)
        if det is not None:
            self._first_det.setdefault(track_id, det)
        if when is not None:
            self._first_when.setdefault(track_id, when)

    def first_seen(self, track_id):
        """Frame number this vehicle was first detected on, or None."""
        return self._first_seen.get(track_id)

    def first_seen_time(self, track_id):
        """Local datetime of this vehicle's first detection, or None."""
        return self._first_when.get(track_id)

    def first_seen_det(self, track_id):
        """Detection dict from this vehicle's first frame, or None."""
        return self._first_det.get(track_id)

    def all_ids(self):
        """Every track ID ever detected (in first-detection order).

        Sourced from the first-seen map, so a track with no usable crop is
        still listed and can get a detection-only fallback record.
        """
        return list(self._first_seen.keys())

    def mark_done(self, track_id):
        """Call this once the vehicle was handed to the analysis step.

        The best crop is kept: a vehicle whose plate could not be read from the
        first crop is retried with this later, better crop before the run ends.
        The candidate crops are dropped - they were only needed to choose the
        crop at submit time.
        """
        self._done.add(track_id)
        self._cand.pop(track_id, None)

    def done_ids(self):
        return self._done

    def __len__(self):
        return len(self._done)

    # ------------------------------------------------ plate-first candidates
    def remember_candidate(self, track_id, crop, score, limit):
        """Keep the best `limit` crops seen for this track, by quality score.

        Offers the Stage-2 worker a small, bounded set of candidate frames so it
        can prefer one where a number plate is visible. Ranked only by the
        EXISTING quality score, so the first plate-visible candidate is also the
        highest-quality one. Nothing is kept when `limit` is 0 or the crop is
        empty.
        """
        if crop is None or limit <= 0:
            return
        candidates = self._cand.setdefault(track_id, [])
        candidates.append((score, crop))
        candidates.sort(key=lambda item: item[0], reverse=True)
        del candidates[limit:]

    def candidate_crops(self, track_id):
        """This track's candidate crops, best quality first (may be empty)."""
        return [crop for _score, crop in self._cand.get(track_id, [])]

    # ------------------------------------------------ best crop per track
    def remember(self, track_id, det, crop, frame_no, good, score=0.0):
        """Remember the best crop seen for this ID.

        A fully visible ("good") box always beats a cut-off one; between two
        boxes of the same kind, the higher quality score wins (larger area
        breaks a tie). Kept for already-analysed vehicles too, so a missed
        plate can be retried from their best crop.
        """
        self._seen.add(track_id)
        self._last_seen[track_id] = frame_no
        box = det["box"]
        area = (box[2] - box[0]) * (box[3] - box[1])
        best = self._best.get(track_id)
        if best is None:
            self._best[track_id] = (good, score, area, det, crop)
            return
        best_good, best_score, best_area, _det, _crop = best
        better = score > best_score or (score == best_score and area > best_area)
        if (good and not best_good) or (good == best_good and better):
            self._best[track_id] = (good, score, area, det, crop)

    def best_for(self, track_id):
        """Return (det, crop) of the best crop remembered, or (None, None)."""
        best = self._best.get(track_id)
        if best is None:
            return None, None
        return best[3], best[4]

    def best_area(self, track_id):
        """Area of the best crop remembered for this ID (0 when none)."""
        best = self._best.get(track_id)
        return best[2] if best is not None else 0

    # ------------------------------------------------ adaptive observation
    def observe(self, track_id, det, crop, frame_no, good, score, sharp,
                stable_iou=0.80):
        """Record one frame of a tracked vehicle.

        Compares this box with the previous frame's to update the stability
        counter, then stores the latest quality/sharpness, the observation count
        and the best crop (via remember()).
        """
        box = det["box"]
        prev = self._prev_box.get(track_id)
        if prev is not None and box_iou(prev, box) >= stable_iou:
            self._stable_run[track_id] = self._stable_run.get(track_id, 0) + 1
        else:
            self._stable_run[track_id] = 0
        self._prev_box[track_id] = list(box)
        self._last_score[track_id] = score
        self._last_sharp[track_id] = sharp
        self._observed[track_id] = self._observed.get(track_id, 0) + 1
        self.remember(track_id, det, crop, frame_no, good, score)

    def ready(self, track_id, min_observe=3, stable_frames=3, min_sharp=25.0,
              convergence_tol=0.05):
        """True when this track's crop is good enough to submit now.

        Needs enough frames observed, a box that has stayed stable for the last
        few frames, a sharp enough crop, and a quality score that has stopped
        improving (within convergence_tol of the best seen so far). Box geometry
        only suggests a usable detection - it does not prove the whole vehicle
        (or its plate) is visible.
        """
        if self._observed.get(track_id, 0) < min_observe:
            return False
        if self._stable_run.get(track_id, 0) < stable_frames:
            return False
        if self._last_sharp.get(track_id, 0.0) < min_sharp:
            return False
        best = self._best.get(track_id)
        if best is None:
            return False
        best_score = best[1]
        if best_score <= 0.0:
            return True
        return (self._last_score.get(track_id, 0.0)
                >= (1.0 - convergence_tol) * best_score)

    def observed(self, track_id):
        """How many frames this track has been seen for."""
        return self._observed.get(track_id, 0)

    def pending(self):
        """Vehicles that were seen but never analysed -> (track_id, det, crop)."""
        return [(tid, b[3], b[4]) for tid, b in self._best.items()
                if tid not in self._done]

    def expired(self, frame_no):
        """IDs not seen for more than track_timeout frames and not analysed yet."""
        limit = frame_no - self.track_timeout
        return [tid for tid, last in self._last_seen.items()
                if tid not in self._done and last <= limit]

    # ---------------------------------------------------------------- stats
    def seen_ids(self):
        return self._seen
