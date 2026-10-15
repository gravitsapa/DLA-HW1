import editdistance


def calc_cer(target_text, predicted_text) -> float:
    distance = editdistance.eval(target_text, predicted_text)
    n = max(1, len(predicted_text))
    return distance / n


def calc_wer(target_text, predicted_text) -> float:
    distance = editdistance.eval(target_text.split(), predicted_text.split())
    n = max(1, len(predicted_text))
    return distance / n
