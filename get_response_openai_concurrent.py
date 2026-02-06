# -*- coding: utf-8 -*-
"""
Concurrent drop-in replacement for get_response.py that calls OpenAI API (gpt-4o / gpt-4o-mini),
while keeping EXACT data formatting and pipeline outputs the same.

Input:
  data/test_data.jsonl (actually JSON list) + data/character_profiles.json
Output:
  results/generation.jsonl  (JSON list, same structure as original script)

Examples:
  pip install -U openai tqdm python-dotenv
  export OPENAI_API_KEY="..."
  python get_response_openai_concurrent.py --model gpt-4o-mini --workers 16

Optional:
  python get_response_openai_concurrent.py --env_file ../../.env
  python get_response_openai_concurrent.py --base_url https://api.openai.com/v1
"""

import os
import json
import time
import argparse
import random
import threading
from typing import List, Dict, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI

# optional dotenv support
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


# ---- copied/faithful to get_response.py ----
def make_inputs(context: str) -> List[Dict[str, str]]:
    dialogues = context.split("\n")
    inputs = []
    for dial in dialogues:
        role = dial.split("：")[0]
        dial = "：".join(dial.split("：")[1:])
        inputs.append({"from": role, "value": dial})
    return inputs


def concat_messages(
    conversations: List[Dict[str, str]], role: str, system_text: str
) -> Tuple[List[Dict[str, str]], str]:
    """
    Faithful to repo get_response.py:
      - history starts with {"role":"user","content": first_query(system_text)}
      - then assistant first_response ("好的！现在我来扮演...") optionally includes first role utterance
      - then for each role turn i>0: add user query from previous non-role utterance and assistant response as role utterance
      - return (history, query) where query is last non-role utterance
    """
    history: List[Dict[str, str]] = []
    first_query = system_text

    # Apply profile renderer if registered
    rendered_role = role
    if _profile_renderer is not None:
        rendered_role = _profile_renderer(role)

    if conversations[0]["from"] == role:
        first_response = f"好的！现在我来扮演 {rendered_role}。 " + "我首先发话：" + conversations[0]["value"]
    else:
        first_response = f"好的！现在我来扮演 {rendered_role}。 "

    history.append({"role": "user", "content": first_query})
    history.append({"role": "assistant", "content": first_response})

    for i in range(len(conversations)):
        if conversations[i]["from"] == role:
            if i == 0:
                continue
            else:
                assert conversations[i - 1]["from"] != role
            query = f" {conversations[i - 1]['from']}：" + conversations[i - 1]["value"]
            response = f" {conversations[i]['from']}：" + conversations[i]["value"]
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})

    assert conversations[-1]["from"] != role
    query = f" {conversations[-1]['from']}：" + conversations[-1]["value"]
    return history, query


# Global profile renderer (can be set via register_profile_renderer)
_profile_renderer: Optional[Any] = None


def register_profile_renderer(renderer):
    """Register a custom profile renderer function.

    Args:
        renderer: A callable that takes a role name (str) and returns a transformed role name (str)
    """
    global _profile_renderer
    _profile_renderer = renderer


# ---- OpenAI call ----
def openai_chat(
    client: OpenAI,
    model: str,
    history: List[Dict[str, str]],
    query: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
) -> str:
    """
    Mirror model.chat(tokenizer, query, messages):
    we pass `history` as messages, and append the final user query as the last message.
    """
    messages = list(history) + [{"role": "user", "content": query}]

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            # exponential backoff + small jitter
            time.sleep(min(30.0, (1.5 ** attempt) + random.random()))
    raise RuntimeError(f"OpenAI call failed: {last_err!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api_key", default="", help="optional; otherwise use env OPENAI_API_KEY")

    parser.add_argument(
        "--base_url",
        default="",
        help='optional; custom OpenAI-compatible API base URL, e.g. "https://api.openai.com/v1" or your proxy URL',
    )

    parser.add_argument(
        "--env_file",
        default="",
        help='optional; path to .env file (if set, requires python-dotenv)',
    )

    parser.add_argument("--test_data", default="data/test_data.jsonl")
    parser.add_argument("--character_profiles", default="data/character_profiles.json")
    parser.add_argument("--out", default="results/generation.jsonl")

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=0, help="0=all")

    # concurrency + checkpointing
    parser.add_argument("--workers", type=int, default=8, help="number of concurrent workers (threads)")
    parser.add_argument("--save_every", type=int, default=100, help="save partial results every N finished samples (0=off)")

    args = parser.parse_args()

    # load env file into os.environ if requested
    if args.env_file:
        if load_dotenv is None:
            raise RuntimeError("python-dotenv not installed. Install: pip install python-dotenv")
        load_dotenv(dotenv_path=args.env_file, override=False)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    print(f"Using OpenAI api_key: {api_key[:4]}****")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. export it, put it in .env, or pass --api_key")

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "")

    if base_url:
        print(f"Using custom OpenAI base URL: {base_url}")
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    os.makedirs("results", exist_ok=True)

    # Repo uses json.load even though filename is .jsonl
    with open(args.test_data, "r", encoding="utf-8") as f:
        datas = json.load(f)
    with open(args.character_profiles, "r", encoding="utf-8") as f:
        role_infos = json.load(f)

    n = len(datas) if args.max_samples <= 0 else min(len(datas), args.max_samples)

    # Pre-allocate to keep original order
    results: List[Optional[Dict[str, Any]]] = [None] * n
    lock = threading.Lock()
    done_counter = 0

    def worker(i: int) -> Tuple[int, Dict[str, Any]]:
        data = datas[i]
        role = data["role"]
        context = data["context"]

        role_information = role_infos[role]
        role_system = f"""{role_information}
现在请你扮演一个角色扮演专家。请你根据上述信息扮演 {role}进行对话。
"""

        conversations = make_inputs(context)
        history, query = concat_messages(conversations, role, role_system)

        out = dict(data)
        try:
            out["model_output"] = openai_chat(
                client=client,
                model=args.model,
                history=history,
                query=query,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
                max_retries=args.retries,
            )
        except Exception as e:
            out["model_output"] = "ERROR"
            out["error"] = repr(e)

        return i, out

    pbar = tqdm(total=n, desc=f"OpenAI generation ({args.model})", mininterval=0.2)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(worker, i) for i in range(n)]
        for fut in as_completed(futures):
            i, out = fut.result()
            results[i] = out

            with lock:
                done_counter += 1
                pbar.update(1)

                # checkpoint
                if args.save_every > 0 and done_counter % args.save_every == 0:
                    partial = [r for r in results if r is not None]
                    with open(args.out, "w", encoding="utf-8") as f:
                        f.write(json.dumps(partial, ensure_ascii=False, indent=4))

    pbar.close()

    final_results = [r for r in results if r is not None]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json.dumps(final_results, ensure_ascii=False, indent=4))

    print(f"Saved {len(final_results)} records to {args.out}")


if __name__ == "__main__":
    main()


# ---- Programmatic API ----
def run_generation(
    test_data: str,
    character_profiles: str,
    output_file: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout: int = 120,
    retries: int = 6,
    workers: int = 8,
    save_every: int = 100,
    max_samples: int = 0,
    profile_renderer: Optional[Any] = None,
    verbose: bool = True,
):
    """Programmatic interface for running generation.

    Args:
        test_data: Path to test data file
        character_profiles: Path to character profiles JSON
        output_file: Path to output file
        api_key: OpenAI API key (uses env var if None)
        base_url: Custom API base URL
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        timeout: Request timeout
        retries: Number of retries
        workers: Number of concurrent workers
        save_every: Save checkpoint every N samples
        max_samples: Max samples to process (0 = all)
        profile_renderer: Optional function to transform role names
        verbose: Whether to print progress
    """
    global _profile_renderer

    # Register profile renderer if provided
    if profile_renderer is not None:
        _profile_renderer = profile_renderer

    # Get API key
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    if verbose:
        print(f"Using OpenAI api_key: {api_key[:4]}****")

    # Create client
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
    if base_url:
        if verbose:
            print(f"Using custom OpenAI base URL: {base_url}")
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Load data
    with open(test_data, "r", encoding="utf-8") as f:
        datas = json.load(f)
    with open(character_profiles, "r", encoding="utf-8") as f:
        role_infos = json.load(f)

    n = len(datas) if max_samples <= 0 else min(len(datas), max_samples)

    # Pre-allocate to keep original order
    results: List[Optional[Dict[str, Any]]] = [None] * n
    lock = threading.Lock()
    done_counter = 0

    def worker(i: int) -> Tuple[int, Dict[str, Any]]:
        data = datas[i]
        role = data["role"]
        context = data["context"]

        role_information = role_infos[role]
        role_system = f"""{role_information}
现在请你扮演一个角色扮演专家。请你根据上述信息扮演 {role}进行对话。
"""

        conversations = make_inputs(context)
        history, query = concat_messages(conversations, role, role_system)

        out = dict(data)
        try:
            out["model_output"] = openai_chat(
                client=client,
                model=model,
                history=history,
                query=query,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout,
                max_retries=retries,
            )
        except Exception as e:
            out["model_output"] = "ERROR"
            out["error"] = repr(e)

        return i, out

    pbar = tqdm(total=n, desc=f"OpenAI generation ({model})", mininterval=0.2, disable=not verbose)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(worker, i) for i in range(n)]
        for fut in as_completed(futures):
            i, out = fut.result()
            results[i] = out

            with lock:
                done_counter += 1
                pbar.update(1)

                # checkpoint
                if save_every > 0 and done_counter % save_every == 0:
                    partial = [r for r in results if r is not None]
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(json.dumps(partial, ensure_ascii=False, indent=4))

    pbar.close()

    final_results = [r for r in results if r is not None]
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(final_results, ensure_ascii=False, indent=4))

    if verbose:
        print(f"Saved {len(final_results)} records to {output_file}")

