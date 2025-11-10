import pytest
import os
from pathlib import Path
import tempfile
import shutil

@pytest.fixture(scope="session")
def tmp_codebase():
    """Create a temporary directory with sample code files for testing indexing."""
    d = Path(tempfile.mkdtemp())

    # create example files
    (d / "AppDelegate.m").write_text("""
    #import "AppDelegate.h"
    - (void)configureAPIToken {
      // token is read from config/initializer/api_config.rb
    }
    """)

    (d / "config" / "initializer").mkdir(parents=True, exist_ok=True)
    (d / "config" / "initializer" / "api_config.rb").write_text("API_TOKEN = 'PLACEHOLDER'")

    (d / "bluetooth.rb").write_text("def init_bluetooth\n  # initialize bluetooth here\nend")

    (d / "README.md").write_text("This file should be ignored.") # Not in INDEX_EXTENSIONS

    yield d

    # Cleanup
    shutil.rmtree(d)
