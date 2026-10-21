"""Training entry point. Implementation pending; see E2 T2 and T3."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--run-name", required=True)
    parser.parse_args()
    raise NotImplementedError("Implement reproducible training and checkpointing according to E2 T2/T3.")


if __name__ == "__main__":
    main()
