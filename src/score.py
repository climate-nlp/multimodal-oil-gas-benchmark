import math
import argparse
import json
import pandas as pd
from typing import Dict, List


def process_score(score: float) -> float:
    assert 0 <= score <= 1
    decimal_places = 1
    new_score = math.floor(score * 10 ** (2 + decimal_places)) / (10 ** decimal_places)
    assert 0 <= new_score <= 100
    return new_score


def calc_precision_recall_f1(
        golds: List[Dict], preds: List[Dict], pos_label: str = None
) -> Dict[str, float or int]:
    g_tpls, s_tpls = set(), set()

    for gold in golds:
        for label in gold['labels']:
            if pos_label is None:
                if 'video_id' in gold:
                    g_tpls.add((gold['video_id'], label))
                else:
                    g_tpls.add((gold['id'], label))
            elif label == pos_label:
                if 'video_id' in gold:
                    g_tpls.add((gold['video_id'],))
                else:
                    g_tpls.add((gold['id'],))

    for pred in preds:
        for label in pred['labels']:
            if pos_label is None:
                if 'video_id' in pred:
                    s_tpls.add((pred['video_id'], label))
                else:
                    s_tpls.add((pred['id'], label))
            elif label == pos_label:
                if 'video_id' in pred:
                    s_tpls.add((pred['video_id'],))
                else:
                    s_tpls.add((pred['id'],))

    g = len(g_tpls)
    s = len(s_tpls)
    c = len(g_tpls & s_tpls)
    p = c / s if s != 0 else 0.
    r = c / g if g != 0 else 0.
    f = ((2 * p * r) / (p + r)) if p + r != 0 else 0.

    assert 0 <= p <= 1
    assert 0 <= r <= 1
    assert 0 <= f <= 1
    assert 0 <= f <= ((p + r) / 2) + 1e-7

    return {
        'n_gold': g, 'n_system': s, 'n_correct': c,
        'precision': process_score(p), 'recall':  process_score(r), 'f': process_score(f)
    }


def evaluate_prediction_file(gold_file: str, prediction_file: str) -> List[Dict]:
    with open(gold_file, 'r') as f:
        golds = [json.loads(l) for l in f.readlines()]

    with open(prediction_file, 'r') as f:
        preds = [json.loads(l) for l in f.readlines()]

    labels = set()
    for gold in golds:
        labels |= set(gold['labels'])
    labels = sorted(labels)

    result = []

    for label in labels:
        for key, score in calc_precision_recall_f1(golds=golds, preds=preds, pos_label=label).items():
            result.append({
                'label': label,
                'metric': key,
                'value': score,
            })
    for key, score in calc_precision_recall_f1(golds=golds, preds=preds, pos_label=None).items():
        result.append({
            'label': f'All',
            'metric': key,
            'value': score,
        })

    return result


def main(args):
    result = evaluate_prediction_file(gold_file=args.g, prediction_file=args.s)
    if args.o:
        pd.DataFrame(result).to_csv(args.o, index=False)
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        type=str,
        help="The gold jsonl file",
    )
    parser.add_argument(
        "-s",
        type=str,
        help="The system prediction jsonl file",
    )
    parser.add_argument(
        "-o",
        "--o",
        type=str,
        default=None,
        help="The output file path",
    )
    args = parser.parse_args()
    main(args=args)
