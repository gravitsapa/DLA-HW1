# DLA ASR Project

Проект на основе [шаблона](https://github.com/Blinorot/pytorch_project_template/tree/example/asr).

Финальная модель - Conformer + RNN-T, обученная на LibriSpeech train-clean-360. [Демо в Google Colab](https://colab.research.google.com/github/gravitsapa/DLA-HW1/blob/conformer/demo.ipynb)

Установка: Python 3.12, FFmpeg; для GPU - драйвер NVIDIA с поддержкой CUDA 12.6. Linux:

```bash
git clone --branch conformer https://github.com/gravitsapa/DLA-HW1.git
cd DLA-HW1
python3 -m venv .venv
source .venv/bin/activate
sudo apt-get update
sudo apt-get install -y ffmpeg
python -m pip install -r requirements.txt
```

Скачивание финального чекпоинта:

```bash
python download_resources.py --checkpoint-url "https://drive.google.com/uc?id=14eyTUGzR2RIToaYBtO5jCt_Xc45a2lkV" --output saved/model.pth
```

LibriSpeech скачивается автоматически при создании датасета. Для shallow fusion используется предобученная [NoTokenLM-Gen-3.5](https://huggingface.co/omurberaisik/NoTokenLM-Gen-3.5), она скачивается автоматически при создании LM-декодера.

Обучение происходило 20 часов на NVIDIA RTX A6000 на LibriSpeech train-clean-360
[Логи](https://www.comet.com/gravitsapa/dla-asr/gri7bl5uo6eafg7kio4dcuh2okg89079).

Инференс на полных test-clean и test-other с greedy, beam search и beam search с LM:

```bash
python inference.py dataloader.batch_size=32 experiment_name=test_final
```

Пользовательский датасет должен иметь структуру:

```text
dataset/
  audio/
    ID1.wav
    ID2.flac
  transcriptions/
    ID1.txt
    ID2.txt
```

Поддерживаются WAV, FLAC, MP3 и M4A. Папка `transcriptions` необязательна. Если она передана в конфиг, для каждой записи требуется соответствующий TXT.

```bash
python -u inference.py --config-name inference_custom custom_dataset_dir=data/dataset experiment_name=custom
```

Для расчёта метрик непосредственно во время инференса добавьте `datasets.custom.transcription_dir=data/dataset/transcriptions`. Без этого параметра инференс не требует эталонов. Чтобы изменить папку результатов, задайте `inferencer.save_dir=PATH`.

Отдельный расчёт CER и WER:

```bash
python calc_metrics.py --dataset-dir data/dataset --predictions-dir saved/inference/custom/predictions/custom/transcriptions --output saved/custom_metrics.json
```

Модель `ASR` собирается через Hydra-конфиги. Независимо выбираются энкодер `conformer` или `deepspeech2` и голова `ctc` или `rnnt`. Для CTC используется линейная голова; RNN-T включает LSTM Predictor и Joint Network. Голова, функция потерь и тип декодера должны соответствовать друг другу. Например, DeepSpeech2 + CTC:

```bash
python train.py --config-name train model/encoder@model.encoder_factory=deepspeech2 model/head@model.head_factory=ctc model/decoder@model.decoder_factory=ctc loss_function=ctc
```

Conformer + RNN-T:

```bash
python train.py --config-name train model/encoder@model.encoder_factory=conformer model/head@model.head_factory=rnnt model/decoder@model.decoder_factory=rnnt loss_function=rnnt
```

Доступны три метода: `greedy`, `beam_search`, `beam_search_lm`. Для последнего выбирается декодер `ctc_lm` или `rnnt_lm`. Список запускаемых методов задаётся в `inferencer.decode_methods`, итоговый метод в `inferencer.final_decode_method`. Например, инференс только с greedy:

```bash
python inference.py 'inferencer.decode_methods=[greedy]' inferencer.final_decode_method=greedy metrics.inference=[]
```

Воспроизведение финального обучения на RTX A6000:

```bash
export COMET_API_KEY=YOUR_API_KEY python train.py --config-name conformer_rnnt_train360_a6000 experiment_name=conf_s_rnnt_train360_reproduction
```
