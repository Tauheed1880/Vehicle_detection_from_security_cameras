"""
Branch A - draw boxes + IDs on the frame and publish the annotated video.

Input : frame + detections
Output: annotated frame (NumPy array). Optionally piped to FFmpeg, which encodes
        H.264 and publishes it to MediaMTX at rtsp://localhost:8554/annotated.
"""
import logging
import shutil
import subprocess

import cv2

log = logging.getLogger(__name__)

GREEN = (0, 200, 0)      # already analysed
ORANGE = (0, 165, 255)   # waiting to be analysed


def draw_detections(frame, detections, done_ids):
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        colour = GREEN if det["track_id"] in done_ids else ORANGE
        label = f"{det['class_name']} #{det['track_id']}"
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(out, label, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out


class FFmpegPublisher:
    """Pipes raw BGR frames into an FFmpeg process that publishes to RTSP."""

    def __init__(self, rtsp_url, width, height, fps):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("FFmpeg was not found. Install it or set "
                               "PUBLISH_ANNOTATED = False in config.py.")
        self.width, self.height = width, height
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp", "-rtsp_transport", "tcp", rtsp_url,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.alive = True
        log.info("Publishing annotated video to %s", rtsp_url)

    def write(self, frame):
        if not self.alive:
            return
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            log.error("FFmpeg closed the pipe. Is MediaMTX running?")
            self.alive = False

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
