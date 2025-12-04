import argparse
import json
import sys
from typing import List, Tuple

import MeCab

from yomigami import kata_to_hira, viterbi

DEFAULT_DATASET = "multi_reading_corpus_v1.json"
DEFAULT_MODEL = "yomigami_model.json"


def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Could not find file: {path}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Failed to parse JSON from {path}: {exc}")
        sys.exit(1)


def init_tagger() -> MeCab.Tagger:
    try:
        return MeCab.Tagger(
            "-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd"
        )
    except RuntimeError:
        # Fall back to default dictionary configuration if neologd is unavailable.
        return MeCab.Tagger("")


def parse_sentence(
    tagger: MeCab.Tagger, sentence: str
) -> Tuple[List[str], List[str]]:
    observations: List[str] = []
    mecab_readings: List[str] = []

    node = tagger.parseToNode(sentence)
    while node:
        surface = node.surface
        if surface and surface != "EOS":
            observations.append(surface)
            features = node.feature.split(",")
            reading_kata = features[7] if len(features) > 7 else "*"
            reading_hira = (
                kata_to_hira(reading_kata) if reading_kata and reading_kata != "*" else "*"
            )
            mecab_readings.append(reading_hira)
        node = node.next

    return observations, mecab_readings


def evaluate(model, dataset, limit: int, show_examples: int):
    tagger = init_tagger()

    token_correct = 0
    token_total = 0
    sentence_exact = 0
    length_mismatches = 0
    examples = []

    for idx, sample in enumerate(dataset):
        if limit and idx >= limit:
            break

        sentence = sample["sentence"]
        gold_reading = [kata_to_hira(r) for r in sample["reading"]]

        observations, mecab_readings = parse_sentence(tagger, sentence)
        predicted_reading, _ = viterbi(model, observations, mecab_readings)

        # Align lengths; penalize missing tokens by counting them as incorrect.
        max_len = max(len(gold_reading), len(predicted_reading))
        token_total += max_len

        if len(gold_reading) != len(predicted_reading):
            length_mismatches += 1

        correct_tokens = sum(
            1
            for g, p in zip(gold_reading, predicted_reading)
            if g == p
        )
        token_correct += correct_tokens

        all_match = (
            len(gold_reading) == len(predicted_reading)
            and correct_tokens == len(gold_reading)
        )
        if all_match:
            sentence_exact += 1
        elif len(examples) < show_examples:
            examples.append(
                {
                    "sentence": sentence,
                    "gold": gold_reading,
                    "pred": predicted_reading,
                }
            )

    token_accuracy = (token_correct / token_total * 100) if token_total else 0.0
    sentence_accuracy = (
        sentence_exact / min(len(dataset), limit or len(dataset)) * 100
    )

    print("--- Evaluation Summary ---")
    print(f"Samples evaluated: {min(len(dataset), limit or len(dataset))}")
    print(f"Token accuracy:    {token_accuracy:.2f}% ({token_correct}/{token_total})")
    print(
        f"Sentence accuracy: {sentence_accuracy:.2f}% (exact match on tokenized sentences)"
    )
    if length_mismatches:
        print(f"Length mismatches: {length_mismatches} (MeCab tokens vs. provided readings)")

    if examples:
        print("\nExample mismatches:")
        for ex in examples:
            print(f"- Sentence: {ex['sentence']}")
            print(f"  Gold: {ex['gold']}")
            print(f"  Pred: {ex['pred']}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YomiGami model accuracy on a labeled corpus."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Path to evaluation corpus JSON (default: multi_reading_corpus_v1.json)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to trained model JSON (default: yomigami_model.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N samples (default: all)",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of mismatch examples to show (default: 5)",
    )
    args = parser.parse_args()

    model = load_json(args.model)
    dataset = load_json(args.dataset)

    evaluate(model, dataset, args.limit, args.examples)


if __name__ == "__main__":
    main()
