# DLA ASR Project

Проект на основе [шаблона](https://github.com/Blinorot/pytorch_project_template/tree/example/asr).

Финальная модель - Conformer + RNN-T, обученная на LibriSpeech train-clean-360. [Демо в Google Colab](./demo.ipynb)

Установка: Python 3.12, FFmpeg, для GPU драйвер NVIDIA с поддержкой CUDA 12.6. Linux:

```bash
git clone --branch asr https://github.com/gravitsapa/DLA-HW1.git
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

Скачивание логов инференса:
```bash
python download_resources.py --logs-url "https://drive.google.com/uc?id=1Oc2dhWBi75q33cL5Qgi1ah8_c8AxYXBN" --logs-output saved/inference/info.log
```

Всё вместе можно скачать просто через
```bash
python download_resources.py
```

LibriSpeech скачивается автоматически при создании датасета.
Для shallow fusion используется предобученная посимвольная LSTM-модель Flair `news-forward-fast`. Использование LM включается добавлением в конфиг метода `beam_search_lm`, обычные `greedy` и `beam_search` работают без LM.

Обучение происходило 22 часа на NVIDIA RTX A6000 на LibriSpeech train-clean-360
[Логи](https://www.comet.com/gravitsapa/dla-asr/gri7bl5uo6eafg7kio4dcuh2okg89079), [Отчет](https://www.comet.com/gravitsapa/dla-asr/reports/C0b2exn7oHVlwLuRNBCwvYXYA)

Инференс на полных test-clean и test-other с greedy, beam search и beam search с LM:

```bash
python inference.py experiment_name=test_eval
```

Результаты будут в `saved\inference\test_eval`

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

Так можно запустить на своем датасете:

```bash
python inference.py --config-name inference_custom custom_dataset_dir=data/dataset experiment_name=custom
```
Результаты будут в `saved\inference\custom`

Также для любого эксперимента можно настраивать набор аудиоаугментаций с различными гиперпараметрами
* Gain - изменение громкости аудио
* SpeedPerturbation - изменение скорости ауди
* AddReverb - добавление реверберации
* AddColoredNoise - добавление шума

Примеры работы всех аугментаций можно посмотреть в этих [логах](https://www.comet.com/gravitsapa/dla-asr/fa77a018b5bc4b928d0897fa3a38cce8)

Для расчёта метрик непосредственно во время инференса добавьте `datasets.custom.transcription_dir=data/dataset/transcriptions`. Без этого параметра инференс не требует эталонов. Чтобы изменить папку результатов, задайте `inferencer.save_dir=PATH`.

Отдельный расчёт CER и WER:

```bash
python calc_metrics.py --dataset-dir data/dataset --predictions-dir saved/inference/custom/predictions/custom/transcriptions --output saved/custom_metrics.json
```

Обучение финальной модели conformer + rnnt:

```bash
export COMET_API_KEY=YOUR_API_KEY
python train.py
```

Модель `ASR` собирается через конфиги. Независимо выбираются энкодер `conformer` или `deepspeech2` и голова `ctc` или `rnnt`. Для CTC используется линейная голова, а RNN-T включает LSTM Predictor и Joint Network. Голова, функция потерь и тип декодера должны соответствовать друг другу. Например, DeepSpeech2 + CTC:

```bash
export COMET_API_KEY=YOUR_API_KEY
python train.py --config-name train model/encoder@model.encoder_factory=deepspeech2 model/head@model.head_factory=ctc model/decoder@model.decoder_factory=ctc loss_function=ctc
```

Доступны три метода: `greedy`, `beam_search`, `beam_search_lm`. Для последнего в конфигах выбирается декодер `ctc_lm` или `rnnt_lm`. Список запускаемых методов задаётся в `inferencer.decode_methods`, итоговый метод в `inferencer.final_decode_method`. Например, инференс только с greedy без подсчета метрик:

```bash
python inference.py 'inferencer.decode_methods=[greedy]' inferencer.final_decode_method=greedy metrics.inference=[]
```
