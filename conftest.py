"""Test configuration for ssres_toolbox."""

import os

# Force CPU for all tests
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"
