# -*- coding: utf-8 -*-
"""NeuroSLM - neuroanatomically inspired small language model."""
__version__ = "0.1.0"

# Fail fast with a helpful message if running on Python older than 3.8.
# This check is written to be parse-safe under Python 2 so users who accidentally
# run `python ...` with system Python 2 get a friendly error rather than a
# confusing SyntaxError from Python-3-only code elsewhere in the package.
try:
	import sys
	if getattr(sys, 'version_info', (0,)) < (3, 8):
		raise ImportError(
			"neuroslm requires Python 3.8 or newer. Please run using your virtualenv\n"
			"python interpreter (e.g. .venv\\Scripts\\python.exe) or 'py -3' on Windows."
		)
except Exception:
	# If sys is missing for some reason, allow import to continue and let
	# subsequent code raise a more specific error later. This keeps import
	# robust in constrained environments.
	pass
