import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

try:
    from requests.exceptions import RequestException
except ImportError:  # pragma: no cover

    class RequestException(Exception):
        pass


def _dump_diagnostics(repo_id, revision, exc):
    print("=" * 60)
    print("Hugging Face download FAILED - diagnostics")
    print("=" * 60)
    print(f"exception type : {type(exc).__name__}")
    print(f"exception msg  : {exc}")
    print(f"repo_id        : {repo_id}")
    print(f"revision       : {revision}")
    print(f"HF_HOME        : {os.environ.get('HF_HOME')}")
    print(f"HF_HUB_CACHE   : {os.environ.get('HF_HUB_CACHE')}")
    print(f"HF_TOKEN set?  : {'yes' if os.environ.get('HF_TOKEN') else 'no'}")
    cache_dir = Path(os.environ.get("HF_HUB_CACHE", ""))
    if cache_dir.exists():
        print(f"cache dir      : {cache_dir}")
        children = sorted(cache_dir.iterdir())
        for child in children[:20]:
            print(f"  - {child.name}")
        if len(children) > 20:
            print(f"  ... ({len(children) - 20} more entries)")
    else:
        print("cache dir      : <does not exist>")
    try:
        import requests

        r = requests.get("https://huggingface.co", timeout=10)
        print(f"huggingface.co : HTTP {r.status_code} (reachable)")
    except Exception as ce:
        print(f"huggingface.co : UNREACHABLE ({type(ce).__name__}: {ce})")
    print("=" * 60)


def snapshot_download_with_retry(*args, max_retries=3, **kwargs):
    repo_id = args[0] if args else kwargs.get("repo_id")
    revision = kwargs.get("revision")
    for attempt in range(max_retries):
        try:
            return snapshot_download(*args, **kwargs)
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None) if e.response else None
            if status == 429 and attempt < max_retries - 1:
                wait_time = 60 * (attempt + 1)
                print(
                    f"Rate limited (429). Waiting {wait_time}s before "
                    f"retry {attempt + 2}/{max_retries}"
                )
                time.sleep(wait_time)
                continue
            _dump_diagnostics(repo_id, revision, e)
            raise
        except RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                print(
                    f"Network error ({type(e).__name__}). Waiting {wait_time}s "
                    f"before retry {attempt + 2}/{max_retries}: {e}"
                )
                time.sleep(wait_time)
                continue
            _dump_diagnostics(repo_id, revision, e)
            raise
        except Exception as e:
            _dump_diagnostics(repo_id, revision, e)
            raise
