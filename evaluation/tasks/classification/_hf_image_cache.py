import os
import time
import urllib.error
import urllib.request

HF_DATASET_BASE_URL = "https://huggingface.co/datasets/yueliu1999/GuardReasoner-VLTest/resolve/main"


def ensure_cached_image(
    image_root: str,
    remote_relative_path: str,
    local_filename: str | None = None,
    timeout_sec: int = 30,
    max_retries: int = 3,
) -> str | None:
    os.makedirs(image_root, exist_ok=True)
    normalized_remote = remote_relative_path.lstrip("/")
    filename = local_filename or os.path.basename(normalized_remote)
    if not filename:
        return None
    local_path = os.path.join(image_root, filename)
    if os.path.isfile(local_path):
        return local_path

    remote_url = f"{HF_DATASET_BASE_URL}/{normalized_remote}"
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(remote_url, timeout=timeout_sec) as response:
                if response.status != 200:
                    raise urllib.error.HTTPError(
                        remote_url, response.status, "Non-200 response", response.headers, None
                    )
                with open(local_path, "wb") as out:
                    out.write(response.read())
            return local_path
        except Exception:
            if attempt == max_retries:
                return None
            time.sleep(min(2**attempt, 8))

    return None
