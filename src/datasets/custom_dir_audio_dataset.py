from pathlib import Path

import soundfile as sf
import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        self._has_ground_truth = transcription_dir is not None
        if self._has_ground_truth and not Path(transcription_dir).is_dir():
            raise ValueError("incorrect transcription_dir")
        data = []
        audio_ids = set()
        for path in sorted(Path(audio_dir).iterdir()):
            if not path.is_file() or path.suffix.lower() not in {
                ".mp3",
                ".wav",
                ".flac",
                ".m4a",
            }:
                continue
            if path.stem in audio_ids:
                raise ValueError(f"Duplicate utterance ID: {path.stem}")
            audio_ids.add(path.stem)

            try:
                info = sf.info(str(path))
                duration = info.frames / info.samplerate
            except (RuntimeError, sf.LibsndfileError):
                audio, sample_rate = torchaudio.load(str(path))
                duration = audio.shape[-1] / sample_rate

            entry = {"path": str(path), "audio_len": duration}
            if self.has_ground_truth():
                transc_path = Path(transcription_dir) / f"{path.stem}.txt"
                if not transc_path.is_file():
                    raise FileNotFoundError(
                        f"wrong reference transcription: {transc_path}"
                    )
                entry["text"] = transc_path.read_text(encoding="utf-8").strip()
            data.append(entry)

        super().__init__(data, *args, **kwargs)

    def has_ground_truth(self) -> bool:
        return self._has_ground_truth
