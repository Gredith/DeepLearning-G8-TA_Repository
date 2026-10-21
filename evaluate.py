"""Evaluation entry point. Implementation pending; see E2 T2 and T6."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.parse_args()
    raise NotImplementedError("Implement frame/video metrics and JSON output according to E2 T2/T6.")


if __name__ == "__main__":
    main()
