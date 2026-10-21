"""Prepare video-level train/validation/test splits. Implementation pending."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.parse_args()
    raise NotImplementedError("Create verified official FF++ video-level splits; see E2 T1.")


if __name__ == "__main__":
    main()
