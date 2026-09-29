"""Command-line entry point: `synth-curator <config.yaml>`.

Builds a SynthCurator from the config and runs it - the literal top-level starting point for
someone who has real images, a synthetic pool, and a config file, and wants to run the whole
pipeline without wiring the Python API themselves.

    $ synth-curator config.yaml                     # -> curated manifest CSV path
    $ synth-curator config.yaml --train-production   # -> also trains the production model
    $ python -m synth_curator config.yaml

With a config `resample:` section, both of the above sweep N seeds (SynthCurator.resample())
instead of running once - see config.example.yaml.

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
    parser.add_argument("--train-production", action="store_true",
                        help="also train the production model on the curated set (needs your "
                             "model's trainer to implement train_production); prints whatever it returns")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")

    sc = SynthCurator.from_config(args.config)
    seeds = SynthCurator.resample_seeds_from_config(args.config)
    if seeds is None:
        print(sc.train_production() if args.train_production else sc.run())
    else:
        for seed, resampled in sc.resample(seeds).items():
            result = resampled.train_production() if args.train_production else resampled.run()
            print(f"seed={seed}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
