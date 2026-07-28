from pathlib import Path


def pytest_configure(config):
    Path("tests/tmp_outputs").mkdir(exist_ok=True)
