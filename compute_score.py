import json
from typing import Dict


def compute_scores(
    evaluation_file: str,
    verbose: bool = True
) -> Dict[str, float]:
    """Compute average scores for each metric.

    Args:
        evaluation_file: Path to evaluation.jsonl
        verbose: Whether to print scores

    Returns:
        Dictionary mapping metric names to average scores
    """
    score_dict = {}
    with open(evaluation_file, "r") as f:
        records = json.load(f)

    for record in records:
        if record['metric_en'] not in score_dict:
            score_dict[record['metric_en']] = []
        score_dict[record['metric_en']].append(record[record['metric_en']])

    result = {}
    for metric in score_dict:
        avg_score = sum(score_dict[metric]) / len(score_dict[metric])
        result[metric] = avg_score
        if verbose:
            print(metric)
            print(avg_score)

    return result


if __name__ == "__main__":
    # Default behavior for backward compatibility
    compute_scores("results/evaluation.jsonl", verbose=True)
