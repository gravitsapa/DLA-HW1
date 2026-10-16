import editdistance


def refined_division(numerator, denominator):
    if denominator == 0:
        if numerator == 0:
            return 0.0
        return float("nan")
    return numerator / denominator


def calc_cer_fraction(target_text, predicted_text) -> dict:
    distance = editdistance.eval(target_text, predicted_text)
    n = len(target_text)
    return {
        "numerator": distance,
        "denominator": n,
    }


def calc_cer(target_text, predicted_text) -> float:
    fraction = calc_cer_fraction(target_text, predicted_text)
    return refined_division(**fraction)


def calc_wer_fraction(target_text, predicted_text) -> dict:
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    distance = editdistance.eval(target_words, predicted_words)
    n = len(target_words)
    return {
        "numerator": distance,
        "denominator": n,
    }


def calc_wer(target_text, predicted_text) -> float:
    fraction = calc_wer_fraction(target_text, predicted_text)
    return refined_division(**fraction)
