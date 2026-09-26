"""Command-line entry point: `synth-curator <config.yaml>`.

Builds a SynthCurator from the config and runs it - the literal top-level starting point for
someone who has real images, a synthetic pool, and a config file, and wants to run the whole
pipeline without wiring the Python API themselves.

    $ synth-curator config.yaml
    $ python -m synth_curator config.yaml

The model comes from the config's `model:` block (a dotted `trainer` path); this entry point has
no way to pass a Python Model object directly - use SynthCurator.from_config in your own script
for that.
"""
from __future__ import annotations

import argparse
import logging

from .pipeline import SynthCurator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-curator")
    parser.add_argument("config", help="path to a config.example.yaml-style YAML file")
    parser.add_argument("-q", "--quiet", action="store_true", help="only show warnings and errors")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")

    manifest = SynthCurator.from_config(args.config).run()
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
