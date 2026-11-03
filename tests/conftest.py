"""Pytest session setup.

On this macOS/conda environment RDKit, PyTorch and scikit-learn each ship their
own libomp, and importing more than one aborts with the OpenMP
"already initialized" error. Allow the duplicate load before any heavy import so
the suite runs deterministically. Set at conftest import time (pytest loads
conftest before collecting test modules) so it precedes torch's libomp load.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
