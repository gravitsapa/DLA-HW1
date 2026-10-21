import argparse
from pathlib import Path

import gdown


def download_checkpoint(url, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = output.with_name(output.name + ".part")
    if not gdown.download(url, str(temporary)):
        raise RuntimeError("Cannot download")
    if not temporary.is_file():
        raise RuntimeError("Not a file")
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-url", required=True)
    parser.add_argument("--output", type=Path, default=Path("saved/model.pth"))
    args = parser.parse_args()

    download_checkpoint(args.checkpoint_url, args.output)


if __name__ == "__main__":
    main()
