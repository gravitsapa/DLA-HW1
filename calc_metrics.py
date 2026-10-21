import argparse
import json
import math
from pathlib import Path

from src.metrics.utils import calc_cer_fraction, calc_wer_fraction, refined_division
from src.text_encoder import TextEncoder


def transcription_files(dir):
    dir = Path(dir)

    return {path.stem: path for path in dir.glob("*.txt") if path.is_file()}


def calc_metrics(ground_truth_dir, predictions_dir):
    references = transcription_files(ground_truth_dir)
    predictions = transcription_files(predictions_dir)

    missing = sorted(references.keys() - predictions.keys())
    extra = sorted(predictions.keys() - references.keys())
    if missing or extra:
        raise ValueError()

    totals = {"CER": [0, 0], "WER": [0, 0]}
    for utterance_id in sorted(references):
        target = TextEncoder.normalize_text(
            references[utterance_id].read_text(encoding="utf-8").strip()
        )
        prediction = TextEncoder.normalize_text(
            predictions[utterance_id].read_text(encoding="utf-8").strip()
        )
        for name, func in (("CER", calc_cer_fraction), ("WER", calc_wer_fraction)):
            fraction = func(target, prediction)
            totals[name][0] += fraction["numerator"]
            totals[name][1] += fraction["denominator"]

    res = {"n_utterances": len(references)}
    for name, (numerator, denominator) in totals.items():
        value = refined_division(numerator, denominator)
        res[name] = None if math.isnan(value) else value
    return res


def main():
    parser = argparse.ArgumentParser()
    references = parser.add_mutually_exclusive_group(required=True)
    references.add_argument(
        "--dataset-dir", type=Path, help="Dataset containing transcriptions/"
    )
    references.add_argument(
        "--ground-truth-dir", type=Path, help="Reference .txt directory directly"
    )
    parser.add_argument(
        "--predictions-dir", required=True, type=Path, help="Predicted .txt directory"
    )
    parser.add_argument("--output", type=Path, help="Optional output JSON file")
    args = parser.parse_args()
    ground_truth_dir = (
        args.dataset_dir / "transcriptions"
        if args.dataset_dir is not None
        else args.ground_truth_dir
    )
    result = calc_metrics(ground_truth_dir, args.predictions_dir)

    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
