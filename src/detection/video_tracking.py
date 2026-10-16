"""Frame-to-frame tracking and temporal class voting for video detection.

A single frame may flicker between classes (pothole vs manhole) or miss an
object entirely. Tracking links detections of the same physical object across
frames; its final class is the confidence-weighted vote over every frame it
was seen in, and short-lived tracks are reported as unconfirmed noise.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.data.dataset_utils import CLASS_NAMES
from src.detection.ground_truth import box_iou


@dataclass
class Track:
    track_id: int
    xyxy: list[int]
    first_frame: int
    last_frame: int
    hits: int = 0
    missed: int = 0
    class_scores: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    best_confidence: float = 0.0

    def add(self, detection: dict, frame_index: int) -> None:
        self.xyxy = list(detection["xyxy"])
        self.last_frame = frame_index
        self.hits += 1
        self.missed = 0
        self.class_scores[detection["class_id"]] += detection["confidence"]
        self.best_confidence = max(self.best_confidence, detection["confidence"])

    @property
    def voted_class_id(self) -> int:
        return max(self.class_scores.items(), key=lambda item: (item[1], -item[0]))[0]

    def summary(self, fps: float | None) -> dict:
        class_id = self.voted_class_id
        total = sum(self.class_scores.values())
        result = {
            "track_id": self.track_id, "class_id": class_id, "class_name": CLASS_NAMES[class_id],
            "frames_seen": self.hits, "first_frame": self.first_frame, "last_frame": self.last_frame,
            "mean_confidence": round(total / self.hits, 4) if self.hits else 0.0,
            "best_confidence": round(self.best_confidence, 4),
            "class_votes": {CLASS_NAMES[cid]: round(score / total, 4) for cid, score in sorted(self.class_scores.items())},
            "last_xyxy": self.xyxy,
        }
        if fps:
            result["first_seen_s"] = round(self.first_frame / fps, 3)
            result["last_seen_s"] = round(self.last_frame / fps, 3)
        return result


class IoUTracker:
    """Greedy class-agnostic IoU tracker; deterministic and dependency-free."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5):
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[dict], frame_index: int) -> list[dict]:
        """Assign ``track_id`` and the track's current voted class to each detection."""
        candidates = sorted(
            ((box_iou(track.xyxy, detection["xyxy"]), track_index, detection_index)
             for track_index, track in enumerate(self.active)
             for detection_index, detection in enumerate(detections)),
            reverse=True,
        )
        matched_tracks: set[int] = set()
        assignment: dict[int, Track] = {}
        for iou, track_index, detection_index in candidates:
            if iou < self.iou_threshold:
                break
            if track_index in matched_tracks or detection_index in assignment:
                continue
            matched_tracks.add(track_index)
            assignment[detection_index] = self.active[track_index]

        for track_index, track in enumerate(self.active):
            if track_index not in matched_tracks:
                track.missed += 1
        for detection_index, detection in enumerate(detections):
            track = assignment.get(detection_index)
            if track is None:
                track = Track(self._next_id, list(detection["xyxy"]), frame_index, frame_index)
                self._next_id += 1
                self.active.append(track)
                assignment[detection_index] = track
            track.add(detection, frame_index)

        still_active = []
        for track in self.active:
            (self.finished if track.missed > self.max_missed else still_active).append(track)
        self.active = still_active

        tracked = []
        for detection_index, detection in enumerate(detections):
            track = assignment[detection_index]
            voted = track.voted_class_id
            tracked.append({**detection, "track_id": track.track_id,
                            "track_class_id": voted, "track_class_name": CLASS_NAMES[voted]})
        return tracked

    def all_tracks(self) -> list[Track]:
        return sorted(self.finished + self.active, key=lambda track: track.track_id)


def summarize_tracks(tracker: IoUTracker, min_frames: int = 3, fps: float | None = None) -> dict:
    """Split tracks into confirmed objects and short-lived noise, counted per class."""
    confirmed, unconfirmed = [], []
    for track in tracker.all_tracks():
        (confirmed if track.hits >= min_frames else unconfirmed).append(track.summary(fps))
    counts = {name: 0 for name in CLASS_NAMES.values()}
    for track in confirmed:
        counts[track["class_name"]] += 1
    return {"min_track_frames": min_frames, "unique_objects_by_class": counts,
            "confirmed_tracks": confirmed, "unconfirmed_track_count": len(unconfirmed)}
