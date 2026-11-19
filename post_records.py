"""
records.json ko ingest API par bhejne wala uploader.

Do tareeqe se chalta hai:
  * standalone:  python post_records.py --file records.json --url http://...
  * integrated:  main.py isay background thread me chalata hai, taake pipeline
                 aur uploader ek hi Docker process me saath chalein.

Behaviour:
  * naya record aate hi POST (records.json ko poll karta hai)
  * jo bhej diya usay dobara nahi bhejta (track_id yaad rakhta hai)
  * server down ho to record "sent" mark nahi hota; connection wapas aane par
    wahi se resume (short exponential backoff, max 30s)
  * bheji hui list `ingest_state.json` me atomic save hoti hai, is liye restart
    par bhi dobara shuru se nahi bhejta

API-first: pipeline ka har naya record pehle `send_now()` ke zariye SEEDHA API
par jata hai; uske baad records.json me backup ke taur par likha jata hai. Jo
record direct POST na ho paye wo file me rehta hai aur yahan background sweep
usay retry karta hai.

Docker: endpoint `INGEST_URL` env var se set karein. Container ke andar
localhost host machine ko nahi milta - `host.docker.internal` ya LAN IP use
karein.
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import requests

log = logging.getLogger("ingest")

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_URL = os.environ.get("INGEST_URL", "http://10.1.19.78:3000/api/detections/ingest")
DEFAULT_KEY = os.environ.get("INGEST_API_KEY", "N1UoJkQ1LWn7wPZmR1v3NofdTHbcn0Sx")
DEFAULT_INTERVAL = float(os.environ.get("INGEST_INTERVAL", "1.0"))
DEFAULT_STATE = os.environ.get("INGEST_STATE", str(BASE_DIR / "ingest_state.json"))

MAX_BACKOFF = 30.0  # is se zyada der tak connection dobara try nahi karega


def load_state(path):
    """(sent, failed) sets wapas laata hai; file na ho to khali."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return set(), set()
    return set(data.get("sent", [])), set(data.get("failed", []))


def save_state(path, sent, failed):
    """Atomic write: adhoori state file kabhi nahi banegi."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"sent": sorted(sent), "failed": sorted(failed)}),
                   encoding="utf-8")
    tmp.replace(path)


def read_records(path):
    """records.json padhta hai; file abhi na ho / adhoori ho to []."""
    try:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return records if isinstance(records, list) else []


def post_record(url, headers, record, timeout):
    """Ek record bhejta hai. (ok, status, text, transient) return karta hai."""
    try:
        resp = requests.post(
            url,
            headers=headers,
            data=json.dumps(record, ensure_ascii=False),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return False, None, str(exc), True

    # 429 / 5xx = thodi der ka masla (retry). 400/401 jaisa = permanent.
    transient = resp.status_code == 429 or resp.status_code >= 500
    return resp.ok, resp.status_code, resp.text, transient


class Uploader:
    """records.json ko tail karke naye records ingest API par POST karta hai."""

    def __init__(self, records_file, url=DEFAULT_URL, api_key=DEFAULT_KEY,
                 state_file=DEFAULT_STATE, interval=DEFAULT_INTERVAL,
                 timeout=15.0, reset_state=False):
        self.records_file = str(records_file)
        self.url = url
        self.state_file = Path(state_file)
        self.interval = interval
        self.timeout = timeout
        self.headers = {"x-api-key": api_key, "Content-Type": "application/json"}
        # Purane run ke "bhej diya" IDs bhulane ke liye: records.json reset hone
        # par ByteTrack IDs dobara 1 se shuru hoti hain, warna uploader naye
        # records ko "already sent" samajh kar chhod deta.
        if reset_state and self.state_file.exists():
            self.state_file.unlink()
        self.sent, self.failed = load_state(self.state_file)
        self._backoff = interval
        self._stop = threading.Event()
        self._thread = None
        # Direct (API-first) path aur background sweep dono isi state ko chhoote
        # hain, is liye ek lock + in-flight guard zaroori hai taake ek hi record
        # do baar POST na ho.
        self._lock = threading.Lock()
        self._inflight = set()
        self.direct_timeout = min(timeout, 3.0)   # worker thread ko zyada na roko

    # ---------------------------------------- integrated (pipeline / Docker)
    def start(self):
        """Background daemon thread me chalu kare."""
        self._thread = threading.Thread(target=self._loop, name="ingest", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout=15.0):
        """Loop rokta hai; bache hue records ek aakhri baar bhejne ki koshish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------- standalone (CLI)
    def run_forever(self):
        """Foreground mode: Ctrl+C tak chalta hai."""
        try:
            self._loop()
        except KeyboardInterrupt:
            self._stop.set()
        log.info("Uploader band. %d bheje, %d reject.", len(self.sent), len(self.failed))

    # ------------------------------------------------------------- internals
    def _save(self):
        save_state(self.state_file, self.sent, self.failed)

    def _claim(self, track_id):
        """Post karne ka haq maangta hai. False = pehle bheja ja chuka / in-flight."""
        with self._lock:
            if (track_id in self.sent or track_id in self.failed
                    or track_id in self._inflight):
                return False
            self._inflight.add(track_id)
            return True

    def _finish(self, track_id, ok, permanent=False):
        """Claim khatam; kamyabi/permanent-reject yaad rakhta hai (transient par kuch nahi)."""
        with self._lock:
            self._inflight.discard(track_id)
            if ok:
                self.sent.add(track_id)
            elif permanent:
                self.failed.add(track_id)
            else:
                return                    # transient: file se baad me retry hoga
            self._save()

    def send_now(self, record, timeout=None):
        """Ek record foran POST kare - API-first path (worker isay call karta hai).

        True wapas aata hai jab record bhej diya gaya (ya pehle hi handle ho chuka
        tha). Transient failure par False - record records.json me rehta hai aur
        background sweep usay dobara bhejta hai.
        """
        track_id = record.get("track_id")
        if not self._claim(track_id):
            return True
        ok, status, text, transient = post_record(
            self.url, self.headers, record,
            timeout if timeout is not None else self.direct_timeout)
        self._finish(track_id, ok, permanent=(not ok and not transient))
        if ok:
            self._backoff = self.interval
            log.info("[OK/direct] track_id=%s -> %s", track_id, status)
        elif transient:
            log.warning("[direct-fail] track_id=%s: %s (file se retry hoga)",
                        track_id, text)
        else:
            log.error("[REJECT/direct] track_id=%s -> %s: %s",
                      track_id, status, str(text)[:200])
        return ok

    def _loop(self):
        log.info("Uploader chalu: %s -> %s (state: %s)",
                 self.records_file, self.url, self.state_file)
        while not self._stop.is_set():
            if self._sweep():
                continue                 # kuch bheja: foran dobara check karo
            if self._stop.wait(self.interval):
                break
        if self._stop.is_set():
            self._sweep(final=True)      # shutdown: bache hue bhej do

    def _sweep(self, final=False):
        """Ek pass. True = kam az kam ek record bhej diya."""
        progressed = False
        for record in read_records(self.records_file):
            if self._stop.is_set() and not final:
                break
            track_id = record.get("track_id")
            if not self._claim(track_id):
                continue

            ok, status, text, transient = post_record(
                self.url, self.headers, record, self.timeout)
            self._finish(track_id, ok, permanent=(not ok and not transient))

            if ok:
                self._backoff = self.interval
                progressed = True
                log.info("[OK] track_id=%s -> %s", track_id, status)
            elif transient:
                if final:
                    log.warning("[SKIP] track_id=%s: server abhi bhi down (%s)",
                                track_id, text)
                    break
                log.warning("[RETRY] track_id=%s: %s (%.0fs baad dobara)",
                            track_id, text, self._backoff)
                time.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, MAX_BACKOFF)
                break
            else:
                # 400/401: dobara bhejne ka faida nahi; record records.json me safe hai
                log.error("[REJECT] track_id=%s -> %s: %s", track_id, status, text[:200])
        return progressed


def parse_args():
    p = argparse.ArgumentParser(description="Live-POST records.json to the ingest API")
    p.add_argument("--file", default=str(BASE_DIR / "records.json"), help="records JSON file")
    p.add_argument("--url", default=DEFAULT_URL, help="ingest endpoint")
    p.add_argument("--key", default=DEFAULT_KEY, help="x-api-key header value")
    p.add_argument("--timeout", type=float, default=15.0, help="per-request timeout (seconds)")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="poll interval (seconds)")
    p.add_argument("--state", default=DEFAULT_STATE, help="state file (sent track_ids)")
    p.add_argument("--reset-state", action="store_true", help="purani yaad clear kare")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uploader = Uploader(args.file, url=args.url, api_key=args.key, state_file=args.state,
                        interval=args.interval, timeout=args.timeout,
                        reset_state=args.reset_state)
    uploader.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
