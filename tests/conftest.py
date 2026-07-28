from pathlib import Path

import pytest


def pytest_configure(config):
    Path("tests/tmp_outputs").mkdir(exist_ok=True)


@pytest.fixture(scope="session", autouse=True)
def check_hf_auth():
    try:
        from huggingface_hub import whoami

        who = whoami()
        print(f"[hf-auth] Authenticated as: {who['name']}")
    except Exception as e:
        print(
            f"[hf-auth] NOT authenticated ({type(e).__name__}: {e}). "
            "Unauthenticated requests hit stricter rate limits."
        )
