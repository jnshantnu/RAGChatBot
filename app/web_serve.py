"""Production entrypoint for the plain HTML/JS frontend (app/web/).
Mirrors app/serve.py's approach for the old Streamlit app: starts loading
the reranker model the moment the process starts, in a background thread,
so the first visitor after a restart doesn't sit through a ~30s wait --
rerank.py's loader is lock-guarded, so a visitor who arrives mid-load just
waits for the remainder rather than triggering a second load.

app/serve.py (Streamlit) is left in place, unused, as a fallback: to roll
back, point the systemd unit's ExecStart at it again and revert the Caddy
route's `uri strip_prefix` (see the commit that introduced this file).
"""
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, so both
sys.path.insert(0, ROOT)  # "app.web.server" (below) and that module's own "from chat import ..." resolve


def _preload():
    try:
        t0 = time.perf_counter()
        from retrieval.rerank import TUNED_MAX_LENGTH, _get_model

        model = _get_model(TUNED_MAX_LENGTH)
        model.predict([("warm up", "warm up")])
        print(f"reranker model preloaded in {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as exc:  # a failed preload must never take the server down
        print(f"reranker preload failed ({exc!r}); the first request will trigger the load instead", flush=True)


if __name__ == "__main__":
    threading.Thread(target=_preload, name="reranker-preload", daemon=True).start()

    import uvicorn

    uvicorn.run("app.web.server:app", host="127.0.0.1", port=8501, log_level="info")
