# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Save outputs *next to this script*, regardless of where you launch Python
import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")  # <project_root>/data

DEFAULT_LOOKBACK_YEARS = 5  # how many years of history to pull
BATCH_SIZE = 50             # tickers per Yahoo batch (avoid throttling)
REQUEST_PAUSE = 2           # seconds between batches (API‑friendly)

pd.options.mode.copy_on_write = True  # pandas 2 speed‑up, safe for our usage
