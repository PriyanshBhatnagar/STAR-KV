"""Train a model to achieve adaptive low-rank KV cache compression.

Uses headwise decomposition for K and joint decomposition for V, with learnable
soft-threshold mechanism that finds optimal ranks during training.

Training runs in two phases, then fuses:
  Phase 1: KD loss + compression loss, alpha LR active.
           Ends when the desired KV compression budget is reached (see
           --desired-comp-rate), with --alpha-samples as a step-count fallback.
  Phase 2 (remaining steps): KD loss only for recovery.
  Fusion:  Sigma is baked into V and reduced to its binary keep-mask, then the
           result is saved to --output.  This is the ONLY artifact produced --
           use it for both accuracy eval and latency benchmarking.
  Phase 3 (optional, --phase3-samples > 0): extra KD-only fine-tune of the
           already-fused model, overwriting --output.  Fusion is numerically
           exact, so this is pure additional recovery, not a correction.

Compression budget
------------------
  --desired-comp-rate C  sets the overall KV cache compression over the COMPRESSED
  layers only -- skipped layers appear in neither numerator nor denominator, so C=0.75
  with --skip-layers 0 1 2 31 means ~75% across the remaining 28 layers.

  V is more sensitive than K, so it keeps more rank: K targets C+delta, V targets
  C-delta, with delta scheduled so the size-weighted overall is exactly C.
      C = 0.60  ->  K removes 70%, V removes 50%   (delta 0.10)
      C = 0.75  ->  K removes 80%, V removes 70%   (delta 0.05)
  Linear between those anchors, flat outside; override with --kv-split-offset.

Example
-------
  python train.py --model meta-llama/Llama-3.1-8B-Instruct --output fused_weights.pt --epochs 1 --lr 2e-5 --seq-len 4096 --num-samples 3500 --alpha-lr 1e-2 --alpha-samples 2500 --comp-weight-k 0.1 --comp-weight-v 0.1 --kd-weight 1.0 --desired-comp-rate 0.6
  python train.py --model meta-llama/Llama-3.2-3B  --output fused_weights.pt --epochs 1 --lr 2e-5 --seq-len 4096 --num-samples 4000 --alpha-lr 1e-2 --alpha-samples 3000 --comp-weight-k 0.1 --comp-weight-v 0.1 --kd-weight 1.0 --desired-comp-rate 0.6
  """


import argparse
import gc
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import login
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    get_scheduler,
)
from tqdm import tqdm

from model import (
    RANK_TILE,
    enforce_rank_floor,
    fuse_and_prune,
    replace_linear_layer,
    collect_K_parameter_size,
    collect_V_parameter_size,
    DecomposeLinear_headwise,
    collect_K_cache_size,
    collect_V_cache_size,
    full_K_cache_size,
    full_V_cache_size,
    freeze_ranks_at_multiple,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Streaming dataset
# ---------------------------------------------------------------------------

class CausalLMBlocks(IterableDataset):
    """Stream a HuggingFace IterableDataset as fixed-length token blocks."""

    def __init__(self, hf_iterable, tokenizer, block_size=1024, max_blocks=None,
                 insert_eos=True):
        self.ds = hf_iterable
        self.tok = tokenizer
        self.block = block_size
        self.max_blocks = max_blocks
        self.insert_eos = insert_eos
        self.eos_id = tokenizer.eos_token_id

    def __iter__(self):
        buf, produced = [], 0
        for ex in self.ds:
            text = ex.get("text") or ex.get("raw_content") or ""
            if not text:
                continue
            ids = self.tok(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            if self.insert_eos and self.eos_id is not None:
                ids = ids + [self.eos_id]
            buf.extend(ids)
            while len(buf) >= self.block:
                chunk = buf[: self.block]
                del buf[: self.block]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(self.block, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long),
                }
                produced += 1
                if self.max_blocks is not None and produced >= self.max_blocks:
                    return

    def __len__(self):
        if self.max_blocks is None:
            raise TypeError("Length unknown; set max_blocks.")
        return self.max_blocks


# ---------------------------------------------------------------------------
# Compression-budget utilities
# ---------------------------------------------------------------------------

def comp_loss(model):
    alphas = [p for n, p in model.named_parameters() if "alpha" in n]
    return torch.sum(torch.stack([torch.exp(-a.cpu()) for a in alphas]))


def comp_loss_k(model):
    alphas = [p for n, p in model.named_parameters() if "k_proj" in n and "alpha" in n]
    return torch.sum(torch.stack([torch.exp(-a.cpu()) for a in alphas]))


def comp_loss_v(model):
    alphas = [p for n, p in model.named_parameters() if "v_proj" in n and "alpha" in n]
    return torch.sum(torch.stack([torch.exp(-a.cpu()) for a in alphas]))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train low-rank KV cache compression via KD.")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="HuggingFace model name or local path")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                   help="HuggingFace dataset name for training")
    p.add_argument("--dataset-config", default="sample-10BT",
                   help="Dataset configuration/subset name (e.g. 'sample-10BT' for fineweb-edu)")
    p.add_argument("--output", default="fused_weights.pt",
                   help="Path to save the final FUSED checkpoint (Sigma baked into V). "
                        "This is the only artifact produced; use it for both eval and latency.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate for non-alpha params")
    p.add_argument("--alpha-lr", type=float, default=1e-2,
                   help="Learning rate for alpha (threshold) params during phase 1")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--num-samples", type=int, default=4000,
                   help="Total training blocks (phase1 + phase2)")
    p.add_argument("--alpha-samples", type=int, default=3000,
                   help="Maximum blocks in phase 1 (step-count fallback); phase 2 "
                        "starts earlier if --desired-comp-rate budget is reached first")
    p.add_argument("--desired-comp-rate", type=float, default=0.6,
                   help="Overall KV cache compression over the COMPRESSED layers only "
                        "(skipped layers are in neither numerator nor denominator). "
                        "0.6 means 60%% removed, so 40%% of full-rank capacity remains. "
                        "K targets rate+delta and V targets rate-delta so the size-weighted "
                        "overall is exactly this value: 0.60 -> K 70%%/V 50%%, "
                        "0.75 -> K 80%%/V 70%% (see --kv-split-offset). "
                        "Phase 2 starts once both targets are met or --alpha-samples is exhausted.")
    p.add_argument("--kd-weight", type=float, default=1.0,
                   help="Weight for the knowledge-distillation KL loss")
    p.add_argument("--comp-weight-k", type=float, default=0.1,
                   help="Weight for the K compression regularisation loss (phase 1 only)")
    p.add_argument("--comp-weight-v", type=float, default=0.1,
                   help="Weight for the V compression regularisation loss (phase 1 only)")
    p.add_argument("--min-head-rank", type=int, default=8,
                   help="Minimum surviving rank per attention head. Stops the "
                        "threshold from collapsing a head to rank 1-4, where it "
                        "stops carrying signal. Usually free: the K cache pads "
                        "every head to its layer's max rank. 0 disables.")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 2, 31],
                   help="Attention layer indices to leave uncompressed")
    p.add_argument("--log-steps", type=int, default=100,
                   help="Print training stats and save checkpoint every N steps")
    p.add_argument("--kv-split-offset", type=float, default=-1.0,
                   help="How far K and V targets sit either side of --desired-comp-rate: "
                        "K targets C+offset, V targets C-offset, so the size-weighted "
                        "overall stays exactly C. V is the more sensitive projection and "
                        "keeps more rank. Default (-1) interpolates the anchors "
                        "C=0.60 -> K 70%%/V 50%% (offset 0.10) and C=0.75 -> K 80%%/V 70%% "
                        "(offset 0.05), held flat outside that range.")
    p.add_argument("--rank-multiple", type=int, default=0,
                   help="Before phase 2, round every K head's surviving rank UP to a "
                        "multiple of this, by lowering its alpha. The decode kernel "
                        "walks the rank dim in BLOCK_SIZE_R=16 tiles and masks the "
                        "tail, so 16 adds directions inside tiles already being "
                        "issued: +22%% K directions for +0%% kernel tiles on "
                        "Llama-3.2-3B, i.e. free in LATENCY. Not free in memory: "
                        "it costs ~2.5 points of compression. K only. 0 (default) "
                        "disables. Phase 2 then recovers "
                        "into the restored directions.")
    p.add_argument("--rank-multiple-v", type=int, default=0,
                   help="Same rounding for the V rank. NOT free the way K is: V is a "
                        "single global factorisation consumed by a plain probs@V GEMM, "
                        "so no tile-masking is already paying for the surplus. It may "
                        "still help that GEMM (a rank_v of 10 runs a 16-wide tile "
                        "anyway), but it is a straight memory trade. 0 (default) leaves "
                        "V alone.")
    p.add_argument("--budget-basis", default="pre-level",
                   choices=["pre-level", "post-level"],
                   help="Which ranks --desired-comp-rate is measured against. "
                        "'pre-level' (default): the raw phase-1 ranks, BEFORE the "
                        "phase-2 round-up to --rank-multiple. Phase 1 stops at the "
                        "target and the round-up is then spent on accuracy, so the "
                        "fused checkpoint ends up LESS compressed than the number you "
                        "asked for (~6 points on Llama-3.2-3B at multiple 16). "
                        "'post-level': measure the rounded ranks, so the fused "
                        "checkpoint lands exactly on --desired-comp-rate and phase 1 "
                        "has to overshoot to get there. Either way the 'after freeze' "
                        "line reports the TRUE achieved compression -- quote that one, "
                        "not the target.")
    p.add_argument("--comp-metric", default="cache", choices=["cache", "legacy"],
                   help="What --desired-comp-rate is measured against. 'cache' (default) "
                        "counts KV CACHE elements per token (rank vs out_features) -- "
                        "the quantity that actually ships. 'legacy' reproduces the old "
                        "behaviour, which counted projection WEIGHT parameters "
                        "((in+out)*rank vs in*out); that differs from the cache by "
                        "(in+out)/in, i.e. 2.00x for MHA and 1.33x for GQA, and "
                        "saturates at 0%% until rank < in*out/(in+out). Both numbers "
                        "are printed either way.")
    p.add_argument("--phase3-samples", type=int, default=0,
                   help="Optional KD-only fine-tune steps AFTER Sigma is fused into V. "
                        "0 (default) skips it; fusion is exact, so this is pure extra "
                        "recovery, not a correction. Result overwrites --output.")
    p.add_argument("--wandb-project", default=None,
                   help="Weights & Biases project name (omit to disable W&B)")
    p.add_argument("--cuda-devices", default="0,1",
                   help="CUDA_VISIBLE_DEVICES string (e.g. '0,1')")
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

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
        wandb_run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
        )

    # ── Load student model ───────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        use_cache=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
    )
    config = model.config

    # ── Load teacher model ───────────────────────────────────────────────────
    print("Loading teacher model...")
    teacher = LlamaForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        use_cache=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── Inject decomposed projections ────────────────────────────────────────
    print("Injecting low-rank decompositions...")
    model = model.float()
    replace_linear_layer(model, config, skip_layers=tuple(args.skip_layers))
    model = model.bfloat16()

    # ── Compression budget (compute full-rank baselines before any training) ─
    # At init all singular values > alpha=0, so equal=True returns in*out for each layer.
    # comp_rate is the fraction REMOVED: 0.7 = 70% compressed, only 30% of full dim remains.
    # Skipped layers are decomposed by neither replace_linear_layer nor counted
    # by the collectors, so they appear in NEITHER numerator nor denominator:
    # a 75% target over the compressed layers prints 75%, not less.
    _n_compressed = len(model.model.layers) - len(set(args.skip_layers))
    _mult = max(args.rank_multiple, 1)
    _mult_v = max(args.rank_multiple_v, 1)

    # Which ranks the phase-1 budget is checked against. "pre-level" measures the
    # raw ranks, so the phase-2 round-up lands on top of the target and the final
    # checkpoint is less compressed than requested; "post-level" measures the
    # rounded ranks, so the final checkpoint lands exactly on the target.
    _bud_k = 1 if args.budget_basis == "pre-level" else _mult
    _bud_v = 1 if args.budget_basis == "pre-level" else _mult_v

    if args.comp_metric == "cache":
        # KV cache elements per token -- the quantity that actually ships.
        full_k_params = full_K_cache_size(model)
        full_v_params = full_V_cache_size(model)
        # The export stores every K head at its rank rounded up to RANK_TILE,
        # so a budget measured below that granularity would stop at a size the
        # checkpoint cannot have. Round the measure up to the tile; for K,
        # --budget-basis then only changes anything when --rank-multiple is
        # coarser than the tile.
        _meas_mult_k = max(_bud_k, RANK_TILE)
        meas_k = lambda m: collect_K_cache_size(m, _meas_mult_k)
        meas_v = lambda m: collect_V_cache_size(m, _bud_v)
    else:
        # Legacy: projection WEIGHT parameters, (in+out)*rank vs in*out.
        full_k_params = collect_K_parameter_size(model, equal=True)
        full_v_params = collect_V_parameter_size(model, equal=True)
        meas_k = lambda m: collect_K_parameter_size(m, equal=True)
        meas_v = lambda m: collect_V_parameter_size(m, equal=True)

    # Reference totals for the report, independent of which metric drives the budget.
    _fk_cache, _fv_cache = full_K_cache_size(model), full_V_cache_size(model)
    _fk_w = collect_K_parameter_size(model, equal=True)
    _fv_w = collect_V_parameter_size(model, equal=True)

    def _ratio_in_out(m):
        """(in+out)/in for the K projection: 2.00 for MHA, ~1.33 for GQA."""
        for name, mod in m.named_modules():
            if isinstance(mod, DecomposeLinear_headwise) and "k_proj" in name:
                return (mod.in_features + mod.out_features) / mod.in_features
        return float("nan")

    def comp_report(m, mk: int = 1, mv: int = 1):
        """Both accountings. Defaults to mk=mv=1, i.e. what the model HAS
        right now -- after the phase-2 freeze the ranks are already rounded, so
        this is the true achieved compression. Pass mk=_mult to project where
        the round-up will land while phase 1 is still running."""
        # What the export actually allocates: per-head widths, each rounded up to
        # RANK_TILE. This is the honest headline.
        ks = collect_K_cache_size(m, max(mk, RANK_TILE))
        v = collect_V_cache_size(m, mv)
        kw, vw = collect_K_parameter_size(m, equal=True), collect_V_parameter_size(m, equal=True)
        return (
            f"cache K={1-ks/_fk_cache:.2%} V={1-v/_fv_cache:.2%} "
            f"KV={1-(ks+v)/(_fk_cache+_fv_cache):.2%} | "
            f"legacy-weights K={1-kw/_fk_w:.2%} V={1-vw/_fv_w:.2%}"
        )

    # ── Split the overall budget between K and V ─────────────────────────────
    # V is the more sensitive projection, so it keeps more rank: K targets
    # C + delta, V targets C - delta. Anchors:
    #     C = 0.60  ->  K 70% / V 50%   (delta = 0.10)
    #     C = 0.75  ->  K 80% / V 70%   (delta = 0.05)
    # Linear between them, held flat outside. --kv-split-offset overrides.
    C = args.desired_comp_rate
    if args.kv_split_offset >= 0.0:
        delta = args.kv_split_offset
    else:
        (c_lo, d_lo), (c_hi, d_hi) = (0.60, 0.10), (0.75, 0.05)
        if C <= c_lo:
            delta = d_lo
        elif C >= c_hi:
            delta = d_hi
        else:
            delta = d_lo + (C - c_lo) * (d_hi - d_lo) / (c_hi - c_lo)

    # Solve for the two rates so the SIZE-WEIGHTED overall is exactly C:
    #     full_k*(1-c_K) + full_v*(1-c_V) == (1-C)*(full_k + full_v)
    #     c_K - c_V                       == 2*delta
    # =>  c_K = C + 2*delta*full_v/W,  c_V = C - 2*delta*full_k/W
    # k_proj and v_proj share out_features on every Llama variant, so full_k ==
    # full_v and this reduces to C +/- delta; the weighted form keeps the overall
    # exact if they ever differ. The previous code used unequal offsets
    # (+0.09/-0.11), which put the overall at C - 0.01 rather than at C.
    W = full_k_params + full_v_params
    k_comp_rate = min(C + 2.0 * delta * full_v_params / W, 0.99)
    v_comp_rate = max(C - 2.0 * delta * full_k_params / W, 0.01)
    k_budget = int((1.0 - k_comp_rate) * full_k_params)
    v_budget = int((1.0 - v_comp_rate) * full_v_params)
    # What the two budgets actually imply, after any clamping.
    nominal_overall = 1.0 - (k_budget + v_budget) / W
    _basis = ("KV cache elements/token"
              if args.comp_metric == "cache" else "projection weight params (legacy)")
    _rounding_on = args.rank_multiple > 1 or args.rank_multiple_v > 1
    _when = ("raw phase-1 ranks, BEFORE the phase-2 round-up"
             if args.budget_basis == "pre-level"
             else "rounded ranks, AFTER the phase-2 round-up")
    print(
        f"Compression targets [{_basis}], over the {_n_compressed} compressed layers "
        f"(skipped layers are in neither numerator nor denominator):\n"
        f"  budget basis = {args.budget_basis} ({_when})\n"
        f"  overall = {nominal_overall:.1%} removed  (asked for {C:.1%}, split delta={delta:.3f})\n"
        f"  K       = {k_comp_rate:.1%} removed ({1-k_comp_rate:.1%} remains, "
        f"budget={k_budget:,} of {full_k_params:,})\n"
        f"  V       = {v_comp_rate:.1%} removed ({1-v_comp_rate:.1%} remains, "
        f"budget={v_budget:,} of {full_v_params:,})"
    )
    # The export rounds every K head up to RANK_TILE anyway, so a pre-level
    # budget only sits below the shipped size when the leveling is COARSER than
    # the tile.
    _k_levels_past_measure = args.rank_multiple > RANK_TILE
    if (args.budget_basis == "pre-level" and _rounding_on
            and not (_k_levels_past_measure or args.rank_multiple_v > 1)):
        print(
            f"  NOTE: the export already rounds every K head up to {RANK_TILE}, and\n"
            f"  --rank-multiple {args.rank_multiple} is no coarser, so pre-level and\n"
            f"  post-level measure the same cache here -- the basis makes no difference."
        )
    elif _rounding_on and args.budget_basis == "pre-level":
        print(
            f"  NOTE: with basis=pre-level the round-up to {args.rank_multiple} lands ON TOP of\n"
            f"  this target, so the fused checkpoint will be LESS compressed than {C:.0%}\n"
            f"  (about 6 points lower on Llama-3.2-3B at multiple 16) -- deliberately,\n"
            f"  the slack is spent on accuracy. Quote the 'after freeze' figure, never\n"
            f"  this target. Use --budget-basis post-level to land exactly on {C:.0%}."
        )
    if abs(nominal_overall - C) > 5e-3:
        print(
            f"  WARNING: the K/V split was clamped, so the overall budget is "
            f"{nominal_overall:.1%}, not the {C:.1%} requested. Lower "
            f"--kv-split-offset (currently {delta:.3f}) to restore it."
        )
    if args.comp_metric == "legacy":
        print(
            "  NOTE: 'legacy' counts (in+out)*rank against in*out, which is NOT the KV\n"
            "  cache size (rank against out_features). They differ by (in+out)/in --\n"
            f"  {_ratio_in_out(model):.2f}x here -- "
            "and the legacy metric reads 0% until rank < in*out/(in+out)."
        )
    if args.rank_multiple > 1 or args.rank_multiple_v > 1:
        print(
            f"  Rank leveling ON at phase 2: "
            f"K -> {'multiples of %d' % args.rank_multiple if args.rank_multiple > 1 else 'off'}, "
            f"V -> {'multiples of %d' % args.rank_multiple_v if args.rank_multiple_v > 1 else 'off'}.\n"
            + (f"  The phase-1 budget measures the POST-leveling cache, so "
               f"--desired-comp-rate {args.desired_comp_rate:.2f} is what the leveled "
               f"model actually achieves -- leveling buys accuracy, not a worse "
               f"headline number."
               if args.budget_basis == "post-level" else
               f"  The phase-1 budget measures the PRE-leveling ranks, so the leveled "
               f"model ends up less compressed than "
               f"--desired-comp-rate {args.desired_comp_rate:.2f}.")
        )
    print(f"  At init: {comp_report(model)}")

    # ── Tokenizer & dataset ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    print(f"Loading dataset: {args.dataset} (config: {args.dataset_config})")
    ds = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    train_iterable = CausalLMBlocks(
        ds, tokenizer, block_size=args.seq_len, max_blocks=args.num_samples
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(
        train_iterable, batch_size=args.batch_size, collate_fn=collator
    )

    # ── Optimizer ────────────────────────────────────────────────────────────
    # K and V alpha groups are split so each can be frozen independently when its
    # compression budget is reached. Freezing is done by setting requires_grad=False
    # on the alpha params (not by zeroing the group lr), because the LR scheduler
    # re-applies base_lr*factor to every group each step and would otherwise undo it.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer = torch.optim.AdamW([
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and "alpha" not in n],
            "weight_decay": 0.01, "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "alpha" in n and "k_proj" in n],
            "lr": args.alpha_lr,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "alpha" in n and "v_proj" in n],
            "lr": args.alpha_lr,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and "alpha" not in n],
            "weight_decay": 0.0, "lr": args.lr,
        },
    ])

    # ── Accelerator ──────────────────────────────────────────────────────────
    accelerator = Accelerator()
    model, teacher, train_loader = accelerator.prepare(model, teacher, train_loader)

    num_steps = len(train_loader) * args.epochs
    lr_sched = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_steps
    )

    # ── Training loop ────────────────────────────────────────────────────────
    # Phase 1/2 best-so-far lands here (still unfused); it is fused into
    # args.output once training ends, then deleted. Not a user-facing artifact.
    staging_path = args.output + ".phase12.tmp"
    best_loss = float("inf")
    phase2_entered = False
    k_frozen = False  # K budget reached; K comp loss dropped
    v_frozen = False  # V budget reached; V comp loss dropped
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            global_step = epoch * len(train_loader) + step

            with torch.no_grad():
                t_out = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                t_logits = t_out.logits

            s_out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            s_logits = s_out.logits

            shift_s = s_logits[:, :-1, :].contiguous()
            shift_t = t_logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()

            mask = shift_labels != tokenizer.pad_token_id
            kd_loss = F.kl_div(
                F.log_softmax(shift_s, dim=-1)[mask],
                F.softmax(shift_t, dim=-1)[mask],
                reduction="batchmean",
            )
            loss = args.kd_weight * kd_loss

            # Phase 1: per-projection comp loss; dropped individually once budget met
            c_loss_k = torch.tensor(0.0)
            c_loss_v = torch.tensor(0.0)
            if not phase2_entered:
                if not k_frozen:
                    c_loss_k = comp_loss_k(model)
                if not v_frozen:
                    c_loss_v = comp_loss_v(model)
                loss = loss + args.comp_weight_k * c_loss_k + args.comp_weight_v * c_loss_v

            accelerator.backward(loss)

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()
                # Re-apply the floor after every alpha update, while alpha is
                # still trainable (phase 2 freezes it, so the rank is settled).
                if not phase2_entered:
                    enforce_rank_floor(
                        accelerator.unwrap_model(model),
                        args.min_head_rank, args.skip_layers,
                    )

                # ── Per-projection budget check; phase 2 when both done ────
                if not phase2_entered:
                    pk = meas_k(model)
                    pv = meas_v(model)

                    if not k_frozen and pk <= k_budget:
                        k_frozen = True
                        # Freeze K alpha (the soft-threshold) via requires_grad=False,
                        # NOT lr=0. The LR scheduler re-applies base_lr*factor to every
                        # group on each step, so a manual lr=0 is overwritten and alpha
                        # keeps training — KD then drives the threshold DOWN, letting
                        # singular values re-cross it and silently decompressing K.
                        raw_model_k = accelerator.unwrap_model(model)
                        for n, p in raw_model_k.named_parameters():
                            if "k_proj" in n and "alpha" in n:
                                p.requires_grad_(False)
                        pbar.write(
                            f"  [K frozen] step={global_step + 1}: "
                            f"K_comp={1-pk/full_k_params:.1%}≥{k_comp_rate:.0%} — "
                            f"K alpha frozen (requires_grad=False), K comp loss dropped"
                        )

                    if not v_frozen and pv <= v_budget:
                        v_frozen = True
                        # Same for V: freeze the threshold itself, not its lr.
                        raw_model_v = accelerator.unwrap_model(model)
                        for n, p in raw_model_v.named_parameters():
                            if "v_proj" in n and "alpha" in n:
                                p.requires_grad_(False)
                        pbar.write(
                            f"  [V frozen] step={global_step + 1}: "
                            f"V_comp={1-pv/full_v_params:.1%}≥{v_comp_rate:.0%} — "
                            f"V alpha frozen (requires_grad=False), V comp loss dropped"
                        )

                    if (k_frozen and v_frozen) or (global_step >= args.alpha_samples):
                        phase2_entered = True
                        reason = (
                            "both budgets reached"
                            if k_frozen and v_frozen
                            else f"alpha_samples={args.alpha_samples} step limit"
                        )
                        raw_model = accelerator.unwrap_model(model)
                        # Round each K head's rank up to a tile boundary BEFORE the
                        # alpha freeze, so phase 2 recovers into the restored
                        # directions. Free at inference: the kernel already issues
                        # ceil(r/16) tiles and masks the tail, so the surplus lanes
                        # sit in tiles being issued anyway.
                        # Phase 1's threshold search is over, so retire it: pin
                        # every rank (K rounded up to a tile boundary) and switch
                        # the forward pass to a binary keep mask. Must happen
                        # before recovery -- phase 2 trains U, Sigma and V as
                        # separate factors over exactly these directions.
                        pbar.write(f"  [Phase 2] freezing ranks "
                                   f"(K -> multiples of {max(args.rank_multiple, 1)}, "
                                   f"V -> multiples of {max(args.rank_multiple_v, 1)})")
                        pbar.write(f"    before freeze (raw phase-1 ranks): "
                                   f"{comp_report(raw_model)}")
                        freeze_ranks_at_multiple(
                            raw_model, max(args.rank_multiple, 1), args.skip_layers,
                            v_multiple=max(args.rank_multiple_v, 1),
                        )
                        pbar.write(f"    ACHIEVED (post-freeze, this is what the "
                                   f"checkpoint has): {comp_report(raw_model)}")
                        # Redundant now that the keep masks are pinned (alpha no
                        # longer gates anything), but kept so alpha cannot collect
                        # gradients or weight decay through recovery.
                        for n, p in raw_model.named_parameters():
                            if "alpha" in n:
                                p.requires_grad_(False)
                        # Fresh optimizer over the remaining trainable params (U/V/diag;
                        # alpha is now frozen out). Resets Adam state so Phase 1 momentum
                        # doesn't carry into recovery.
                        optimizer = torch.optim.AdamW([
                            {
                                "params": [
                                    p for n, p in raw_model.named_parameters()
                                    if p.requires_grad
                                    and not any(nd in n for nd in no_decay)
                                ],
                                "weight_decay": 0.01, "lr": 5e-6,
                            },
                            {
                                "params": [
                                    p for n, p in raw_model.named_parameters()
                                    if p.requires_grad
                                    and any(nd in n for nd in no_decay)
                                ],
                                "weight_decay": 0.0, "lr": 5e-6,
                            },
                        ])
                        steps_remaining = max(1, num_steps - global_step)
                        lr_sched = get_scheduler(
                            "linear", optimizer=optimizer,
                            num_warmup_steps=0,
                            num_training_steps=steps_remaining,
                        )
                        pbar.write(
                            f"  [Phase 2] step={global_step + 1}: alpha frozen, fresh "
                            f"optimizer (U/V/diag, {reason}), KD loss only for recovery"
                        )

            if (step + 1) % args.log_steps == 0:
                pk = meas_k(model)
                pv = meas_v(model)
                k_comp = 1.0 - pk / full_k_params
                v_comp = 1.0 - pv / full_v_params
                phase_tag = "phase1" if not phase2_entered else "phase2"
                pbar.write(
                    f"  [{phase_tag}] step={global_step + 1}"
                    f"  loss={loss.item():.4f}"
                    f"  kd={kd_loss.item():.4f}"
                    f"  comp_k={c_loss_k.item():.4f}  comp_v={c_loss_v.item():.4f}"
                    f"  K_comp={k_comp:.2%}(target={k_comp_rate:.0%})"
                    f"  V_comp={v_comp:.2%}(target={v_comp_rate:.0%})"
                )
                _rm = accelerator.unwrap_model(model)
                pbar.write(f"            now:   {comp_report(_rm)}")
                if _rounding_on and not phase2_entered:
                    pbar.write(f"            after round-up it becomes: "
                               f"{comp_report(_rm, _mult, _mult_v)}")
                if wandb_run:
                    wandb_run.log({
                        "loss": loss.item(),
                        "kd_loss": kd_loss.item(),
                        "comp_loss_k": c_loss_k.item(),
                        "comp_loss_v": c_loss_v.item(),
                        "k_comp": k_comp,
                        "v_comp": v_comp,
                        "phase": 1 if not phase2_entered else 2,
                        "step": global_step,
                    })

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), staging_path)
                    pbar.write(f"  Saved checkpoint → {staging_path}")

        print(f"Epoch {epoch + 1} done. Best loss: {best_loss:.4f}")

    print(f"Training complete. Best Phase 1/2 loss: {best_loss:.4f}")

    # ── Fuse Sigma into V, then save the single final artifact ───────────────
    # The in-memory model sits at the LAST step, not the best one, so restore
    # the best Phase 1/2 state before fusing.
    raw_model = accelerator.unwrap_model(model)

    # Free the Phase 1/2 optimizer before fusing. Its Adam moments are sized for
    # the pre-fusion U/Sigma/V parameters, which fusion is about to replace --
    # keeping it alive pins a full copy of exp_avg/exp_avg_sq for tensors the
    # model no longer uses, and that is enough to OOM phase 3 on its first step.
    del optimizer, lr_sched
    gc.collect()
    torch.cuda.empty_cache()

    if os.path.exists(staging_path):
        print(f"Restoring best Phase 1/2 state from {staging_path}...")
        raw_model.load_state_dict(
            torch.load(staging_path, map_location="cpu", weights_only=False), strict=False
        )

    print("Fusing Sigma into V and pruning dead ranks...")
    fuse_and_prune(raw_model, args.skip_layers)
    torch.save(raw_model.state_dict(), args.output)
    print(f"Fused checkpoint → {args.output}")
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 3 (optional): further KD-only fine-tune of the fused model ─────
    if args.phase3_samples > 0:
        print("\nPhase 3: fine-tuning the fused model with KD loss (pure PyTorch, no Triton)...")

        # New optimizer — as old one references now-gone U/S/V params.
        p3_optimizer = torch.optim.AdamW(
            [p for p in raw_model.parameters() if p.requires_grad],
            lr=5e-6,
            weight_decay=0.01,
        )

        # Fresh streaming slice for Phase 3 (restarts from dataset beginning).
        print(f"Loading Phase 3 dataset ({args.phase3_samples} blocks)...")
        ds3 = load_dataset(
            args.dataset,
            name=args.dataset_config,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        p3_loader = DataLoader(
            CausalLMBlocks(ds3, tokenizer, block_size=args.seq_len,
                           max_blocks=args.phase3_samples),
            batch_size=args.batch_size,
            collate_fn=collator,
        )
        p3_loader = accelerator.prepare(p3_loader)

        model.train()
        best_p3_loss = float("inf")
        pbar3 = tqdm(p3_loader, desc="Phase 3 (fused U/VS)")

        for step, batch in enumerate(pbar3):
            with torch.no_grad():
                t_out = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                t_logits = t_out.logits

            s_out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            s_logits = s_out.logits

            shift_s = s_logits[:, :-1, :].contiguous()
            shift_t = t_logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()

            mask = shift_labels != tokenizer.pad_token_id
            kd_loss = F.kl_div(
                F.log_softmax(shift_s, dim=-1)[mask],
                F.softmax(shift_t, dim=-1)[mask],
                reduction="batchmean",
            )
            loss = args.kd_weight * kd_loss

            accelerator.backward(loss)

            if (step + 1) % args.grad_accum == 0:
                p3_optimizer.step()
                p3_optimizer.zero_grad()

            if (step + 1) % args.log_steps == 0:
                pbar3.write(
                    f"  [phase3] step={step + 1}"
                    f"  kd_loss={kd_loss.item():.4f}"
                )
                if wandb_run:
                    wandb_run.log({
                        "phase3_kd_loss": kd_loss.item(),
                        "phase": 3,
                        "step": step,
                    })

            if loss.item() < best_p3_loss:
                best_p3_loss = loss.item()
                torch.save(raw_model.state_dict(), args.output)
                pbar3.write(f"  [phase3] Saved fused checkpoint → {args.output}")

        print(f"Phase 3 done. Best fused loss: {best_p3_loss:.4f}")

    if os.path.exists(staging_path):
        os.remove(staging_path)
    print(f"\nFinal fused checkpoint → {args.output}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
