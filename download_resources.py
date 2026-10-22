import argparse
from pathlib import Path

import gdown


def download_file(url, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = output.with_name(output.name + ".part")
    if not gdown.download(url, str(temporary)):
        raise RuntimeError("Cannot download")
    if not temporary.is_file():
        raise RuntimeError("Not a file")
    temporary.replace(output)
    return output


def download_checkpoint(url, output):
    return download_file(url, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-url",
        default="https://drive.google.com/uc?id=14eyTUGzR2RIToaYBtO5jCt_Xc45a2lkV",
    )
    parser.add_argument("--output", type=Path, default=Path("saved/model.pth"))
    parser.add_argument(
        "--logs-url",
        default="https://drive.google.com/uc?id=1Oc2dhWBi75q33cL5Qgi1ah8_c8AxYXBN",
    )
    parser.add_argument(
        "--logs-output", type=Path, default=Path("saved/inference/info.log")
    )
    args = parser.parse_args()

    if not args.checkpoint_url and not args.logs_url:
        parser.error("Provide --checkpoint-url or --logs-url")
    if args.checkpoint_url:
        download_checkpoint(args.checkpoint_url, args.output)
    if args.logs_url:
        download_file(args.logs_url, args.logs_output)


if __name__ == "__main__":
    main()
