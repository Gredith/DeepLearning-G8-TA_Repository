"""Detect, crop, and resize faces. Detector choice is pending in DECISIONS.md."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.parse_args()
    raise NotImplementedError("Implement face crops and missing-face reporting according to E2 T1.")


if __name__ == "__main__":
    main()
