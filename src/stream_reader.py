"""
Step 6 - OpenCV reads the stream.

Input : a video file path or an RTSP URL (str)
Output: frames, one at a time. Each frame is a NumPy array (H, W, 3), uint8, BGR.
"""
import logging
import time

import cv2

log = logging.getLogger(__name__)

_STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")


def _open(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def probe_stream(source):
    """Return (width, height, fps) of a source. Needed by the FFmpeg publisher."""
    cap = _open(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps != fps or fps <= 0 or fps > 120:   # fps != fps catches NaN
        fps = 25.0
    return width, height, fps


def read_frames(source, reconnect_attempts=5, loop=False):
    """Generator: yields BGR frames until the video ends (or forever for a live stream)."""
    source = str(source)
    is_stream = source.lower().startswith(_STREAM_PREFIXES)
    cap = _open(source)
    failures = 0
    just_rewound = False
    try:
        while True:
            ok, frame = cap.read()
            if ok:
                failures = 0
                just_rewound = False
                yield frame
                continue

            if not is_stream:                       # a normal video file
                if loop and not just_rewound:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    just_rewound = True
                    continue
                log.info("End of video file.")
                return

            failures += 1                           # a live stream that dropped
            if failures > reconnect_attempts:
                raise RuntimeError("Lost the stream and could not reconnect.")
            log.warning("Stream read failed (%d/%d), reconnecting...",
                        failures, reconnect_attempts)
            cap.release()
            time.sleep(2)
            cap = _open(source)
    finally:
        cap.release()
