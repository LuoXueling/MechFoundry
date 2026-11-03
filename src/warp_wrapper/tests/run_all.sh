#!/bin/bash
# Run the complete wrapper test suite. Native Warp FEM precision types can be
# exercised together in one process.
#
# Usage:  conda activate mechfoundry-warp117 && cd src/warp_wrapper && bash tests/run_all.sh

set -euo pipefail

# JAX-AMG shares the GPU with Warp and Torch; avoid JAX reserving most memory.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# JAX-AMG and pyamgx cannot reliably initialize AMGX sequentially in the same
# process. Run the pyamgx preconditioner module in a fresh interpreter.
python -m pytest tests/ -v --ignore=tests/test_preconditioners.py "$@"
python -m pytest tests/test_preconditioners.py -v "$@"
echo "=== All tests passed ==="
