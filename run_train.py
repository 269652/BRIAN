"""
Lightweight launcher for NeurolSLM training that first checks Python version.
This script is intentionally written without modern Python syntax so running it
with an old `python` (like system Python 2) will still execute and print a
helpful message instead of producing a SyntaxError from the training module.

Usage:
  python run_train.py --preset small --steps 2000 --batch_size 8 --device cuda

If you have a virtualenv, prefer:
  C:\path\to\venv\Scripts\python.exe run_train.py ...
or use the PowerShell helper `run_train.ps1` included in the repo.
"""
import sys
import runpy

MIN_PY = (3, 8)
if sys.version_info < MIN_PY:
    sys.stderr.write(
        "neuroslm requires Python %d.%d+ to run.\n" % MIN_PY
    )
    sys.stderr.write("Detected python %s.\n" % (sys.version.split()[0],))
    sys.stderr.write("Please run using your project's virtualenv Python or 'py -3'.\n")
    sys.exit(2)

# If version is OK, run the package entrypoint (this will execute neuroslm.train)
# We forward sys.argv so command-line args are respected by the module.
runpy.run_module('neuroslm.train', run_name='__main__')
