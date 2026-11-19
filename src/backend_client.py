"""
Step 14 - Send the record to the FastAPI backend (Stage 3).

HTTP POST, multipart/form-data with two parts:
  payload  : the record as a JSON string
  snapshot : the JPEG image file
Agree these two names with whoever builds Stage 3.
"""
import json
import logging
import time

import requests

log = logging.getLogger(__name__)


def send_record(record, jpg_bytes, url, timeout=5, retries=2):
    """Returns True if the backend answered with a 2xx status."""
    for attempt in range(1, retries + 2):
        try:
            response = requests.post(
                url,
                data={"payload": json.dumps(record)},
                files={"snapshot": (f"{record['track_id']}.jpg", jpg_bytes, "image/jpeg")},
                timeout=timeout,
            )
            if response.ok:
                log.info("Sent vehicle #%s to backend (%s)", record["track_id"], response.status_code)
                return True
            log.warning("Backend replied %s: %s", response.status_code, response.text[:200])
        except requests.RequestException as exc:
            log.warning("Backend not reachable (attempt %d): %s", attempt, exc)
        time.sleep(1)
    log.error("Giving up on vehicle #%s", record["track_id"])
    return False
