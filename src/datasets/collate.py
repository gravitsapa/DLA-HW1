import torch
import torch.nn.functional as F


def collate_fn(
    dataset_items: list[dict], text_pad_token: int = 0, squeeze_channels: bool = True
):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    if len(dataset_items) == 0:
        return {}

    n_audio_channels = dataset_items[0]["audio"].shape[0]
    n_mel_features = dataset_items[0]["spectrogram"].shape[1]

    n_samples_max = 0
    n_mel_frames_max = 0
    n_tokens_max = 0

    for item in dataset_items:
        n_audio_channels_here, n_samples = item["audio"].shape
        assert n_audio_channels_here == n_audio_channels
        n_samples_max = max(n_samples_max, n_samples)

        n_audio_channels_here, n_mel_features_here, n_mel_frames = item[
            "spectrogram"
        ].shape
        assert n_audio_channels_here == n_audio_channels
        assert n_mel_features == n_mel_features_here
        n_mel_frames_max = max(n_mel_frames_max, n_mel_frames)

        n_audio_channels_here, n_tokens = item["text_encoded"].shape
        assert n_audio_channels_here == n_audio_channels
        n_tokens_max = max(n_tokens_max, n_tokens)

    audio_list = []
    audio_length_list = []
    spectrogram_list = []
    spectrogram_length_list = []
    text_encoded_list = []
    text_encoded_length_list = []
    text_list = []
    audio_path_list = []

    for item in dataset_items:
        audio = item["audio"]
        if squeeze_channels:
            audio = audio.squeeze(0)
        audio_len = audio.shape[-1]
        audio_list.append(F.pad(audio, (0, n_samples_max - audio_len)))
        audio_length_list.append(audio_len)

        spectrogram = item["spectrogram"]
        if squeeze_channels:
            spectrogram = spectrogram.squeeze(0)
        spectrogram_len = spectrogram.shape[-1]
        spectrogram_list.append(
            F.pad(spectrogram, (0, n_mel_frames_max - spectrogram_len))
        )
        spectrogram_length_list.append(spectrogram_len)

        text_encoded = item["text_encoded"]
        if squeeze_channels:
            text_encoded = text_encoded.squeeze(0)
        text_encoded_len = text_encoded.shape[-1]
        text_encoded_list.append(
            F.pad(
                text_encoded,
                (0, n_tokens_max - text_encoded_len),
                value=text_pad_token,
            )
        )
        text_encoded_length_list.append(text_encoded_len)

        text_list.append(item["text"])
        audio_path_list.append(item["audio_path"])

    batch_dim = 0

    return {
        "audio": torch.stack(
            audio_list,
            dim=batch_dim,
        ),
        "audio_length": torch.tensor(audio_length_list, dtype=torch.long),
        "spectrogram": torch.stack(
            spectrogram_list,
            dim=batch_dim,
        ),
        "spectrogram_length": torch.tensor(spectrogram_length_list, dtype=torch.long),
        "text_encoded": torch.stack(
            text_encoded_list,
            dim=batch_dim,
        ),
        "text_encoded_length": torch.tensor(text_encoded_length_list, dtype=torch.long),
        "text": text_list,
        "audio_path": audio_path_list,
    }
