"""Put harness/src (the shared scoring code) on the import path and locate my feature cache.
Every script here imports this first, so it runs from any working directory."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS_SRC = os.path.normpath(os.path.join(HERE, '..', '..', 'harness', 'src'))
CACHE = os.path.normpath(os.path.join(HERE, '..', 'cache'))
for p in (HERE, HARNESS_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)
