import sys
import torch
import json
from typing import Optional


def run_reward_model(
    generation_trans_file: str,
    character_profiles: str,
    output_file: str,
    reward_model_path: str = 'BaichuanCharRM/',
    max_seq_length: int = 4096,
    verbose: bool = True
):
    """Run reward model scoring on generated responses.

    Args:
        generation_trans_file: Path to generation_trans.jsonl
        character_profiles: Path to character_profiles.json
        output_file: Path to output evaluation.jsonl
        reward_model_path: Path to reward model
        max_seq_length: Maximum sequence length
        verbose: Whether to print progress

    Note:
        This requires GPU and the BaichuanCharRM model files.
    """
    try:
        from BaichuanCharRM.modeling_baichuan import BaichuanCharRM
        from BaichuanCharRM.tokenization_baichuan import BaichuanTokenizer
    except ImportError:
        raise RuntimeError(
            "BaichuanCharRM not found. This function requires the reward model to be installed."
        )

    with open(character_profiles, "r") as f:
        character_profile = json.load(f)
    with open(generation_trans_file, mode='r') as f:
        records = json.load(f)

    def format_input(example):
        input_text = "<RoleInfo>\n\n" \
            + str(character_profile[example['role']]) + "\n\n<Context>\n\n" + example['context'] + "\n\n<Response>\n\n" + example['model_output'] + "\n\n<Dimension>\n\n" + example["metric_zh"]
        return input_text

    tokenizer = BaichuanTokenizer.from_pretrained(reward_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Multi-GPU sharding with memory limits
    max_memory = {
        0: "22GiB",
        1: "22GiB",
        2: "22GiB",
        3: "22GiB",
        4: "22GiB",
        "cpu": "64GiB",
    }

    base_model = BaichuanCharRM.from_pretrained(
        reward_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    embed_device = base_model.model.embed_tokens.weight.device

    if verbose:
        import tqdm
        iterator = tqdm.tqdm(records)
    else:
        iterator = records

    for record in iterator:
        input_text = format_input(record)
        input_ids = tokenizer.encode(text=input_text, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[-max_seq_length:]
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(embed_device)
        with torch.inference_mode():
            score = base_model(input_ids=input_ids)[1].item() * 4 + 1
            record[record['metric_en']] = score

    with open(output_file, 'w') as f:
        f.write(json.dumps(records, ensure_ascii=False, indent=4))

    if verbose:
        print(f"Reward model scoring complete: {output_file}")


if __name__ == "__main__":
    # Default behavior for backward compatibility
    run_reward_model(
        generation_trans_file="results/generation_trans.jsonl",
        character_profiles="data/character_profiles.json",
        output_file='results/evaluation.jsonl',
        verbose=True
    )

