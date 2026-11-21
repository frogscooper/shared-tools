import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", help="run tests that hit MAST / ExoFOP / NASA / Gaia")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live") or os.environ.get("SHARED_TOOLS_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live network test; use --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fixtures():
    return FIXTURES
