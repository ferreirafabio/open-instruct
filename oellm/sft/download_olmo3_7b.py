#!/usr/bin/env python3
import argparse
import os
from huggingface_hub import snapshot_download


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download the Olmo-3-1025-7B base checkpoint via Hugging Face snapshot."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="models/Olmo-3-1025-7B",
        help="Directory where the downloaded files will be stored.",
    )
    parser.add_argument(
        "--repo-id",
        default="allenai/Olmo-3-1025-7B",
        help="Hugging Face repo ID for the model.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (defaults to HF_TOKEN env var).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        cache_dir=args.output_dir,
        token=args.token,
    )

    print(f"Downloaded {args.repo_id} into {args.output_dir}")


if __name__ == "__main__":
    main()

