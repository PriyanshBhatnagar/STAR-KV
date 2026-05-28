"""Evaluate a trained low-rank KV cache model.

Supports:
  - Perplexity on WikiText-2, C4, PTB
  - lm-eval tasks: PIQA, WinoGrande, ARC, HellaSwag, OpenBookQA
  - Long-context: LongBench tasks
  - Long-context: RULER tasks

Example
-------
  # Zero-shot accuracy on standard tasks
  python eval.py \\
    --model lmsys/longchat-7b-v1.5-32k \\
    --weights trained_weights.pt \\
    --tasks piqa,winogrande,arc_easy,arc_challenge,openbookqa,hellaswag \\
    --batch-size 32

  # Perplexity
  python eval.py \\
    --model lmsys/longchat-7b-v1.5-32k \\
    --weights trained_weights.pt \\
    --ppl --ppl-datasets wikitext2,c4

  # Long-context benchmarks
  python eval.py \\
    --model lmsys/longchat-7b-v1.5-32k \\
    --weights trained_weights.pt \\
    --longbench --ruler
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from model import (
    replace_linear_layer,
    replace_attn_with_triton,
    set_model_mode,
)


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

def _get_ppl_data(name: str, tokenizer, seqlen: int):
    if "wikitext2" in name:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    if "c4" in name:
        class _Wrap:
            def __init__(self, ids): self.input_ids = ids
        data = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        enc = tokenizer(" ".join(data[:1100]["text"]), return_tensors="pt")
        return _Wrap(enc.input_ids[:, : 256 * seqlen])
    if "ptb" in name:
        data = load_dataset("ptb_text_only", "penn_treebank", split="test")
        return tokenizer("\n\n".join(data["sentence"]), return_tensors="pt")
    raise ValueError(f"Unknown PPL dataset: {name}")


@torch.no_grad()
def evaluate_ppl(model, tokenizer, datasets: str, seqlen: int = 2048, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    results = {}
    for name in datasets.split(","):
        name = name.strip()
        loader = _get_ppl_data(name, tokenizer, seqlen)
        enc = loader.input_ids
        nsamples = enc.numel() // seqlen
        nlls = []
        for i in tqdm(range(nsamples), desc=f"PPL [{name}]"):
            batch = enc[:, i * seqlen : (i + 1) * seqlen].to(device)
            out = model(input_ids=batch, use_cache=False)
            logits = out.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = enc[:, i * seqlen : (i + 1) * seqlen][:, 1:].to(device)
            loss = nn.CrossEntropyLoss()(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            nlls.append(loss.float() * seqlen)
        avg_loss = torch.stack(nlls).sum() / (len(nlls) * seqlen)
        ppl = torch.exp(avg_loss).item()
        results[name] = {"loss": avg_loss.item(), "ppl": ppl}
        print(f"  {name:12s}  seqlen={seqlen}  loss={avg_loss.item():.4f}  ppl={ppl:.2f}")
    return results


# ---------------------------------------------------------------------------
# lm-eval evaluation
# ---------------------------------------------------------------------------

def evaluate_lmeval(model, tokenizer, tasks: str, batch_size: int,
                    max_length: int = None, model_name: str = ""):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    kwargs = {"pretrained": model, "tokenizer": tokenizer, "add_bos_token": False,
              "batch_size": batch_size}
    if max_length is not None:
        kwargs["max_length"] = max_length
        model.config.max_position_embeddings = max_length

    lm_obj = HFLM(**kwargs)
    task_manager = lm_eval.tasks.TaskManager()

    task_list = [t.strip() for t in tasks.split(",")]
    print(f"Running lm-eval tasks: {task_list}")
    with torch.no_grad():
        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=task_list,
            task_manager=task_manager,
            log_samples=False,
        )
    print(make_table(results))
    return results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a low-rank KV cache model.")
    p.add_argument("--model", required=True,
                   help="HuggingFace model name or local path")
    p.add_argument("--weights", required=True,
                   help="Path to trained weights (.pt file from train.py)")
    p.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 31],
                   help="Attention layers left uncompressed during training")

    # Zero-shot accuracy
    p.add_argument("--tasks", default=None,
                   help="Comma-separated lm-eval tasks, e.g. piqa,winogrande,arc_easy")
    p.add_argument("--batch-size", type=int, default=32)

    # Perplexity
    p.add_argument("--ppl", action="store_true", help="Run perplexity evaluation")
    p.add_argument("--ppl-datasets", default="wikitext2,c4",
                   help="Comma-separated PPL datasets: wikitext2, c4, ptb")
    p.add_argument("--ppl-seqlen", type=int, default=2048)

    # Long-context benchmarks
    p.add_argument("--longbench", action="store_true",
                   help="Run LongBench tasks (requires lm_eval[longbench])")
    p.add_argument("--ruler", action="store_true",
                   help="Run RULER tasks (requires lm_eval[ruler])")
    p.add_argument("--long-batch-size", type=int, default=4,
                   help="Batch size for long-context tasks")
    p.add_argument("--max-length", type=int, default=31500,
                   help="Max context length for long-context evaluations")

    p.add_argument("--output", default=None,
                   help="JSON file to write all results (optional)")
    p.add_argument("--cuda-devices", default="0",
                   help="CUDA_VISIBLE_DEVICES string")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)

    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        use_cache=False,
        use_safetensors=True,
    )
    config = model.config

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Inject decomposed structure, load weights ────────────────────────────
    print("Injecting low-rank decompositions...")
    replace_linear_layer(model, config, skip_layers=tuple(args.skip_layers))

    print(f"Loading weights: {args.weights}")
    model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=False),
        strict=False,
    )
    model = model.bfloat16()

    # Export inference-ready weights (fold Sigma, prune dead ranks) and replace
    # attention modules with the unified prefill+decode forward.
    print("Exporting inference weights and replacing attention modules...")
    model = replace_attn_with_triton(
        model, config, skip_layers=tuple(args.skip_layers), dtype=torch.bfloat16
    )
    set_model_mode(model, "triton", skip_layers=tuple(args.skip_layers))
    model.eval()
    model.config.use_cache = True

    all_results = {}

    # ── Perplexity ───────────────────────────────────────────────────────────
    if args.ppl:
        print("\n=== Perplexity Evaluation ===")
        for seqlen in [1024, 2048]:
            res = evaluate_ppl(
                model, tokenizer, args.ppl_datasets, seqlen=seqlen,
                device=get_device()
            )
            all_results[f"ppl_seqlen{seqlen}"] = res

    # ── Zero-shot accuracy ───────────────────────────────────────────────────
    if args.tasks:
        print("\n=== Zero-shot Accuracy ===")
        res = evaluate_lmeval(
            model, tokenizer,
            tasks=args.tasks,
            batch_size=args.batch_size,
            model_name=args.model,
        )
        all_results["lmeval"] = res

    # ── LongBench ────────────────────────────────────────────────────────────
    if args.longbench:
        longbench_tasks = (
            "longbench_hotpotqa,longbench_qasper,longbench_triviaqa,"
            "longbench_multi_news,longbench_trec,longbench_lcc,"
            "longbench_samsum,longbench_narrativeqa,longbench_qmsum,"
            "longbench_vcsum,longbench_dureader"
        )
        print("\n=== LongBench ===")
        res = evaluate_lmeval(
            model, tokenizer,
            tasks=longbench_tasks,
            batch_size=args.long_batch_size,
            max_length=args.max_length,
            model_name=args.model,
        )
        all_results["longbench"] = res

    # ── RULER ────────────────────────────────────────────────────────────────
    if args.ruler:
        print("\n=== RULER ===")
        res = evaluate_lmeval(
            model, tokenizer,
            tasks="ruler",
            batch_size=args.long_batch_size,
            max_length=args.max_length,
            model_name=args.model,
        )
        all_results["ruler"] = res

    # ── Save results ─────────────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
