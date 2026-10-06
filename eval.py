"""Evaluate a fused low-rank KV cache model (the checkpoint train.py --output writes).

Supports:
  - Perplexity on WikiText-2, C4, PTB
  - lm-eval tasks: PIQA, WinoGrande, ARC, HellaSwag, OpenBookQA
  - Long-context: LongBench tasks
  - Long-context: RULER tasks

Pass --triton to evaluate through the fused Triton attention (the path
latency.py benchmarks) rather than the pure-PyTorch reference.

Example
-------
  # Zero-shot accuracy on standard tasks
  python eval.py \\
    --model meta-llama/Llama-3.2-3B \\
    --weights fused_weights.pt \\
    --tasks piqa,winogrande,arc_easy,arc_challenge,openbookqa,hellaswag \\
    --batch-size 32

  # Perplexity
  python eval.py \\
    --model meta-llama/Llama-3.2-3B \\
    --weights fused_weights.pt \\
    --ppl --ppl-datasets wikitext2,c4

  # Baseline perplexity (uncompressed model, no weights needed)
  python eval.py \\
    --model meta-llama/Llama-3.2-3B \\
    --baseline --ppl --ppl-datasets wikitext2,c4

  # Through the deployed Triton attention path
  python eval.py \\
    --model meta-llama/Llama-3.2-3B \\
    --weights fused_weights.pt --triton \\
    --ppl --ppl-datasets wikitext2

  # Long-context benchmarks
  python eval.py \\
    --model meta-llama/Llama-3.2-3B \\
    --weights fused_weights.pt \\
    --longbench --ruler

  # With 4-bit KV quantization on top of the low-rank cache (fp32 fake quant)
  python eval.py \\
    --model meta-llama/Llama-3.2-3B \\
    --weights fused_weights.pt --kv-quant \\
    --tasks piqa,openbookqa
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from model import (
    fold_kv_hadamard,
    load_compressed_checkpoint,
    replace_attn_with_triton,
    set_kv_fake_quant,
    set_model_mode,
)


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

def _get_ppl_data(name: str, tokenizer, seqlen: int):
    if "wikitext2" in name:
        data = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
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
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import make_table

    kwargs = {"pretrained": model, "tokenizer": tokenizer, "add_bos_token": False,
              "batch_size": batch_size}
    if max_length is not None:
        kwargs["max_length"] = max_length
        model.config.max_position_embeddings = max_length

    lm_obj = HFLM(**kwargs)
    task_manager = TaskManager()

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
    p.add_argument("--weights", default=None,
                   help="Path to trained weights (not needed for --baseline)")
    p.add_argument("--baseline", action="store_true",
                   help="Evaluate the uncompressed model (no weights needed)")
    p.add_argument("--kv-quant", action="store_true",
                   help="4-bit KV quantization on top of the low-rank cache. Alone: fp32 "
                        "fake quant on the PyTorch path, every forward (prefill too). "
                        "With --triton: the packed cache and kernels -- exact prefill, "
                        "quantized decode, as deployed.")
    p.add_argument("--triton", action="store_true",
                   help="Evaluate through the fused Triton attention path (what "
                        "latency.py benchmarks) instead of the pure-PyTorch reference")
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
    p.add_argument("--ppl-seqlen", type=int, nargs="+", default=[1024, 2048],
                   help="Context length(s) to score perplexity at; one run per "
                        "value (e.g. --ppl-seqlen 4096, or 1024 2048 4096)")

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
    p.add_argument("--cpu-load", action="store_true",
                   help="Load and decompress on the host, then move the bf16 model "
                        "to the GPU. The checkpoint load runs in fp32 (~27 GB for a "
                        "7B model), which does not fit on a 24 GB card; this keeps "
                        "only the bf16 result resident.")
    p.add_argument("--cuda-devices", default="0",
                   help="CUDA_VISIBLE_DEVICES string")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if not args.baseline and args.weights is None:
        raise ValueError("--weights is required unless --baseline is set")
    if args.kv_quant and args.baseline:
        raise ValueError("--kv-quant quantizes the low-rank cache; it needs --weights, not --baseline")
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
        device_map=None if args.cpu_load else "auto",
        use_cache=False,
        use_safetensors=True,
    )
    config = model.config

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Inject decomposed structure, load weights ────────────────────────────
    if args.baseline:
        model = model.bfloat16()
        skip_layers = tuple(args.skip_layers)
    else:
        print(f"Loading weights: {args.weights}")
        model = model.float()
        # Installs pruned U/VS modules for a fused checkpoint, or the legacy
        # U/Sigma/V ones otherwise, and raises if any factor fails to load.
        # skip_layers comes back as recorded in the checkpoint, so the export
        # below cannot disagree with it about which layers are compressed.
        model, fused, skip_layers = load_compressed_checkpoint(
            model, config, args.weights, skip_layers=tuple(args.skip_layers)
        )
        print(f"  format: {'fused U/VS (pruned)' if fused else 'legacy U/Sigma/V'}"
              f"   compressed layers: all except {skip_layers}")
        if args.kv_quant:
            # In fp32, before the cast: the rotation is folded into VS and U once.
            fold_kv_hadamard(model)
        model = model.bfloat16()

    # Staged on the host: the checkpoint is loaded in fp32, which for a 7B model
    # is ~27 GB and does not fit beside anything else on a 24 GB card. Only the
    # bf16 result needs to be resident, so move it now -- before the export
    # below, which places its tensors on whatever device the layer already sits.
    if args.cpu_load:
        model = model.to(get_device())

    # Optionally export inference-ready weights (fold Sigma, prune dead ranks) and
    # swap in the fused Triton attention -- the same path latency.py benchmarks.
    # Off by default: the pure-PyTorch DecomposeLinear path is the accuracy
    # reference. Note perplexity runs prefill only (use_cache=False), so it does
    # not exercise the decode kernel; --triton matters for generation tasks.
    if args.triton and not args.baseline:
        print("Exporting inference weights and replacing attention modules..."
              + ("  (4-bit packed KV cache)" if args.kv_quant else ""))
        model = replace_attn_with_triton(
            model, config, skip_layers=skip_layers, dtype=torch.bfloat16,
            kv_quant=args.kv_quant,
        )
        set_model_mode(model, "triton", skip_layers=skip_layers)
    elif args.kv_quant:
        print("4-bit KV fake quantization (fp32) on every compressed K/V latent")
        set_kv_fake_quant(model, True)
    model.eval()
    model.config.use_cache = True

    all_results = {}

    # ── Perplexity ───────────────────────────────────────────────────────────
    if args.ppl:
        print("\n=== Perplexity Evaluation ===")
        for seqlen in args.ppl_seqlen:
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
            "longbench_qasper, longbench_multi_news,longbench_trec,longbench_qmsum,longbench_vcsum"
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
