import json
import copy


def run_transform(
    generation_file: str,
    id2metric_file: str,
    output_file: str,
    verbose: bool = True
):
    """Transform generation format to include metrics.

    Args:
        generation_file: Path to generation.jsonl
        id2metric_file: Path to id2metric.jsonl
        output_file: Path to output generation_trans.jsonl
        verbose: Whether to print progress
    """
    with open(id2metric_file, 'r') as f:
        id_metric = json.load(f)

    with open(generation_file, 'r') as f:
        datas = json.load(f)

    results = []

    for data in datas:
        if data['model_output'] is not None and data['model_output'] != "ERROR":
            model_output = data['model_output'].split("\n")[0]  # Prevent continuous generation
            data['model_output'] = model_output
            if str(data['id']) in id_metric:
                for x in id_metric[str(data['id'])]:
                    data['metric_en'] = x[0]
                    data['metric_zh'] = x[1]
                    tmp = copy.deepcopy(data)
                    results.append(tmp)

    with open(output_file, 'w') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))

    if verbose:
        print(f"Transformed {len(results)} records to {output_file}")


if __name__ == "__main__":
    # Default behavior for backward compatibility
    run_transform(
        generation_file='results/generation.jsonl',
        id2metric_file='data/id2metric.jsonl',
        output_file='results/generation_trans.jsonl',
        verbose=True
    )

