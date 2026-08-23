"""Apple-Gorilla (AG): a self-improving prompt executor built on the Claude API.

Pipeline: optimize -> execute -> critique -> iterate.
Self-improvement: propose a patch to AG's own tunable files, snapshot, test,
adopt only if the test suite passes, else auto-rollback.
"""

__version__ = "0.1.0"
