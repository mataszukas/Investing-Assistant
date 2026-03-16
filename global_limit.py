import time
from datetime import datetime, timezone
from typing import Optional, Tuple
from google.cloud import firestore

_db = firestore.Client()


def check_rate_limit_global(
    client_id: str,
    limit: int = 20,
    window_seconds: int = 600,
) -> Tuple[bool, Optional[int]]:
    """
    Global fixed-window rate limit across ALL Cloud Run instances.
    Returns: (allowed, retry_after_seconds)
    """
    now = int(time.time())
    window_id = now // window_seconds
    window_end = (window_id + 1) * window_seconds
    retry_after = max(0, window_end - now)

    doc_id = f"{client_id}:{window_id}"
    doc_ref = _db.collection("rate_limits").document(doc_id)

    expire_at = datetime.fromtimestamp(window_end, tz=timezone.utc)

    doc_ref.set({"count": firestore.Increment(1), "expireAt": expire_at}, merge=True)

    count = int((doc_ref.get().get("count") or 0))
    if count > limit:
        return False, retry_after
    return True, None
