"""Make the src package importable for tests without installing the project."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session", autouse=True)
def _cleanup_streamlit_apptest_tempdir():
    """Clean up AppTest's module-level temp dir so ``-W error`` exits cleanly."""
    yield
    try:
        from streamlit.testing.v1.app_test import TMP_DIR
    except ImportError:
        return
    TMP_DIR.cleanup()
