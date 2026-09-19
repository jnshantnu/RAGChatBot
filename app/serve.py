"""Production entrypoint: starts Streamlit exactly like
`streamlit run app/streamlit_app.py <flags>` (all flags pass straight through),
but begins loading the reranker model the moment the process starts.

Why: Streamlit only runs the app script when a visitor connects, so warming the
model from inside the script (see _warm_reranker in streamlit_app.py) means
the first visitor after every restart sits through a ~30s spinner. Loading in a
background thread from process start means that wait has usually finished
before anyone arrives; a visitor who does arrive mid-load simply waits for the
remainder (rerank.py's loader is lock-guarded, so nothing loads twice).

The model lives in this process's memory, so it is still lost on restart --
this moves the wait, it can't remove it.
"""
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _preload():
    try:
        t0 = time.perf_counter()
        from retrieval.rerank import TUNED_MAX_LENGTH, _get_model

        model = _get_model(TUNED_MAX_LENGTH)
        # The first predict() pays one-time thread-pool/allocator setup; do it
        # here so the first real query doesn't.
        model.predict([("warm up", "warm up")])
        print(f"reranker model preloaded in {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as exc:  # a failed preload must never take the server down; the app's own warm-up retries
        print(f"reranker preload failed ({exc!r}); the first visitor will trigger the load instead", flush=True)


if __name__ == "__main__":
    threading.Thread(target=_preload, name="reranker-preload", daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", os.path.join(ROOT, "app", "streamlit_app.py"), *sys.argv[1:]]
    sys.exit(stcli.main())
