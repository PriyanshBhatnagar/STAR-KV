"""Train a model to achieve adaptive low-rank KV cache compression.

Uses headwise decomposition for K and joint decomposition for V, with learnable
soft-threshold mechanism that finds optimal ranks during training.

Training runs in two phases over --num-samples blocks:
  Phase 1: U, Sigma and V train as separate factors with KD loss + compression
           loss, while the threshold alpha searches for the ranks. Ends when the
           KV compression budget is reached (see --comp-ratio), with
           --alpha-samples as a step-count fallback.
  Phase 2: the ranks are tiled (K up to a multiple of --rank-multiple-k, V of
           --rank-multiple-v) and pinned, Sigma is fused into V and the dead
           directions are pruned -- the model is now in its deployed form -- and
           the remaining blocks are KD-only recovery of that fused model. The
           best fused state is saved to --output: the ONLY artifact, used for
           both accuracy eval and latency benchmarking.

Compression budget
------------------
  --comp-ratio C  sets the overall KV cache compression over the COMPRESSED
  layers only -- skipped layers appear in neither numerator nor denominator. The
  budget deliberately lands COMP_SLACK (1.5 points) under C, leaving a little rank
  for accuracy: C=0.75 with --skip-layers 0 1 2 31 means 73.5% across the
  remaining 28 layers, and C=0.60 means 58.5%.

  V is more sensitive than K, so it keeps more rank: K targets (C-slack)+delta, V
  targets (C-slack)-delta, with delta scheduled so the size-weighted overall is
  exactly C-slack.
      C = 0.60  ->  K removes 68.5%, V removes 48.5%   (delta 0.10)
      C = 0.75  ->  K removes 78.5%, V removes 68.5%   (delta 0.05)
  Linear between those anchors, flat outside; override with --kv-split-offset.

Example
-------
  python train.py --model meta-llama/Llama-3.1-8B-Instruct --output fused_weights.pt --epochs 1 --lr 2e-5 --seq-len 4096 --num-samples 3500 --alpha-lr 1e-2 --alpha-samples 2500 --comp-weight-k 0.1 --comp-weight-v 0.1 --kd-weight 1.0 --comp-ratio 0.6
  python train.py --model meta-llama/Llama-3.2-3B  --output fused_weights.pt --epochs 1 --lr 2e-5 --seq-len 4096 --num-samples 4000 --alpha-lr 1e-2 --alpha-samples 3000 --comp-weight-k 0.1 --comp-weight-v 0.1 --kd-weight 1.0 --comp-ratio 0.6
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

# How far under --comp-ratio the budget deliberately lands: 0.75 -> 73.5%,
# 0.60 -> 58.5%. A little rank left for accuracy, applied as one shift of the
# K/V split so the overall stays exact.
COMP_SLACK = 0.015

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
                        "starts earlier if --comp-ratio budget is reached first")
    p.add_argument("--comp-ratio", type=float, default=0.6,
                   help="Overall KV cache compression over the COMPRESSED layers only "
                        "(skipped layers are in neither numerator nor denominator). "
                        "The budget lands 1.5 points under this value, leaving a "
                        "little rank for accuracy: 0.6 -> 58.5%% removed (K 68.5%%/V "
                        "48.5%%), 0.75 -> 73.5%% removed (K 78.5%%/V 68.5%%); see "
                        "--kv-split-offset. Phase 2 starts once both targets are met "
                        "or --alpha-samples is exhausted.")
    p.add_argument("--kd-weight", type=float, default=1.0,
                   help="Weight for the knowledge-distillation KL loss")
    p.add_argument("--comp-weight-k", type=float, default=0.1,
                   help="Weight for the K compression regularisation loss (phase 1 only)")
    p.add_argument("--comp-weight-v", type=float, default=0.1,
                   help="Weight for the V compression regularisation loss (phase 1 only)")
    p.add_argument("--min-head-rank", type=int, default=8,
                   help="Minimum surviving rank per attention head. Stops the "
                        "threshold from collapsing a head to rank 1-4, where it "
                        "stops carrying signal. Every head is stored at its own "
                        "rank, so the floor costs cache memory, and the budget "
                        "counts it. 0 disables.")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 2, 31],
                   help="Attention layer indices to leave uncompressed")
    p.add_argument("--log-steps", type=int, default=100,
                   help="Print training stats and save checkpoint every N steps")
    p.add_argument("--kv-split-offset", type=float, default=-1.0,
                   help="How far K and V targets sit either side of the overall "
                        "target T = --comp-ratio - 0.015: K targets T+offset, V targets "
                        "T-offset, so the size-weighted overall stays exactly T. V is "
                        "the more sensitive projection and keeps more rank. Default (-1) "
                        "interpolates the anchors C=0.60 -> K 68.5%%/V 48.5%% (offset "
                        "0.10) and C=0.75 -> K 78.5%%/V 68.5%% (offset 0.05), held flat "
                        "outside that range.")
    p.add_argument("--rank-multiple-k", type=int, default=0,
                   help="Before phase 2, round every K head's surviving rank UP to a "
                        "multiple of this, by lowering its alpha. The decode kernel "
                        "walks the rank dim in BLOCK_SIZE_R=16 tiles and masks the "
                        "tail, so 16 adds directions inside tiles already being "
                        "issued: +22%% K directions for +0%% kernel tiles on "
                        "Llama-3.2-3B, i.e. free in LATENCY. Not free in memory: "
                        "the budget counts the rounded ranks, so phase 1 compresses "
                        "harder to make room. K only. 0 (default) disables. Phase 2 "
                        "then recovers into the restored directions.")
    p.add_argument("--rank-multiple-v", type=int, default=0,
                   help="Same rounding for the V rank. NOT free the way K is: V is a "
                        "single global factorisation consumed by a plain probs@V GEMM, "
                        "so no tile-masking is already paying for the surplus. It may "
                        "still help that GEMM (a rank_v of 10 runs a 16-wide tile "
                        "anyway), but it is a straight memory trade. 0 (default) leaves "
                        "V alone.")
    p.add_argument("--comp-metric", default="cache", choices=["cache", "legacy"],
                   help="What --comp-ratio is measured against. 'cache' (default) "
                        "counts KV CACHE elements per token (rank vs out_features) -- "
                        "the quantity that actually ships. 'legacy' reproduces the old "
                        "behaviour, which counted projection WEIGHT parameters "
                        "((in+out)*rank vs in*out); that differs from the cache by "
                        "(in+out)/in, i.e. 2.00x for MHA and 1.33x for GQA, and "
                        "saturates at 0%% until rank < in*out/(in+out). Both numbers "
                        "are printed either way.")
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
    _mult = max(args.rank_multiple_k, 1)
    _mult_v = max(args.rank_multiple_v, 1)

    # The phase-1 budget is checked against the ranks as phase 2 will round them
    # (--rank-multiple-k / --rank-multiple-v), so the fused checkpoint lands exactly
    # on --comp-ratio rather than the round-up landing on top of it.

    if args.comp_metric == "cache":
        # KV cache elements per token -- the quantity that actually ships.
        full_k_params = full_K_cache_size(model)
        full_v_params = full_V_cache_size(model)
        # The export stores every K head at its rank rounded up to RANK_TILE,
        # so a budget measured below that granularity would stop at a size the
        # checkpoint cannot have. Round the measure up to the tile.
        _meas_mult_k = max(_mult, RANK_TILE)
        meas_k = lambda m: collect_K_cache_size(m, _meas_mult_k)
        meas_v = lambda m: collect_V_cache_size(m, _mult_v)
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
    # The overall target is T = C - COMP_SLACK. V is the more sensitive
    # projection, so it keeps more rank: K targets T + delta, V targets T - delta.
    # delta is scheduled on the requested C. Anchors:
    #     C = 0.60  ->  T 58.5%: K 68.5% / V 48.5%   (delta = 0.10)
    #     C = 0.75  ->  T 73.5%: K 78.5% / V 68.5%   (delta = 0.05)
    # Linear between them, held flat outside. --kv-split-offset overrides.
    C = args.comp_ratio
    T = C - COMP_SLACK
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

    # Solve for the two rates so the SIZE-WEIGHTED overall is exactly T:
    #     full_k*(1-c_K) + full_v*(1-c_V) == (1-T)*(full_k + full_v)
    #     c_K - c_V                       == 2*delta
    # =>  c_K = T + 2*delta*full_v/W,  c_V = T - 2*delta*full_k/W
    # k_proj and v_proj share out_features on every Llama variant, so full_k ==
    # full_v and this reduces to T +/- delta; the weighted form keeps the overall
    # exact if they ever differ. (The original code relaxed the overall with
    # unequal offsets, +0.09/-0.11; here the relaxation is the single shift
    # COMP_SLACK, so both projections give up the same share.)
    W = full_k_params + full_v_params
    k_comp_rate = min(T + 2.0 * delta * full_v_params / W, 0.99)
    v_comp_rate = max(T - 2.0 * delta * full_k_params / W, 0.01)
    k_budget = int((1.0 - k_comp_rate) * full_k_params)
    v_budget = int((1.0 - v_comp_rate) * full_v_params)
    # What the two budgets actually imply, after any clamping.
    nominal_overall = 1.0 - (k_budget + v_budget) / W
    _basis = ("KV cache elements/token"
              if args.comp_metric == "cache" else "projection weight params (legacy)")
    _rounding_on = args.rank_multiple_k > 1 or args.rank_multiple_v > 1
    print(
        f"Compression targets [{_basis}], over the {_n_compressed} compressed layers "
        f"(skipped layers are in neither numerator nor denominator):\n"
        f"  overall = {nominal_overall:.1%} removed  (--comp-ratio {C:.1%} less "
        f"{COMP_SLACK:.1%} slack, split delta={delta:.3f})\n"
        f"  K       = {k_comp_rate:.1%} removed ({1-k_comp_rate:.1%} remains, "
        f"budget={k_budget:,} of {full_k_params:,})\n"
        f"  V       = {v_comp_rate:.1%} removed ({1-v_comp_rate:.1%} remains, "
        f"budget={v_budget:,} of {full_v_params:,})"
    )
    if abs(nominal_overall - T) > 5e-3:
        print(
            f"  WARNING: the K/V split was clamped, so the overall budget is "
            f"{nominal_overall:.1%}, not the {T:.1%} targeted. Lower "
            f"--kv-split-offset (currently {delta:.3f}) to restore it."
        )
    if args.comp_metric == "legacy":
        print(
            "  NOTE: 'legacy' counts (in+out)*rank against in*out, which is NOT the KV\n"
            "  cache size (rank against out_features). They differ by (in+out)/in --\n"
            f"  {_ratio_in_out(model):.2f}x here -- "
            "and the legacy metric reads 0% until rank < in*out/(in+out)."
        )
    if args.rank_multiple_k > 1 or args.rank_multiple_v > 1:
        print(
            f"  Rank leveling ON at phase 2: "
            f"K -> {'multiples of %d' % args.rank_multiple_k if args.rank_multiple_k > 1 else 'off'}, "
            f"V -> {'multiples of %d' % args.rank_multiple_v if args.rank_multiple_v > 1 else 'off'}.\n"
            f"  The phase-1 budget measures the leveled cache, so {T:.1%} is "
            f"what the leveled model actually achieves -- leveling buys "
            f"accuracy, not a worse headline number."
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
    # Only phase 2 states are saved: they are fused, at the final ranks, i.e.
    # exactly what ships. The best of them (lowest loss at a log step) is what
    # --output holds when training ends.
    best_loss = float("inf")
    saved = False
    phase2_entered = False
    k_frozen = False  # K budget reached; K comp loss dropped
    v_frozen = False  # V budget reached; V comp loss dropped
    model.train()

    def enter_phase2(write):
        """Tile and pin the ranks, then fuse: from here on the model is in its
        deployed form, and recovery trains exactly what ships."""
        raw = accelerator.unwrap_model(model)
        write(f"  [Phase 2] freezing ranks (K -> multiples of {_mult}, "
              f"V -> multiples of {_mult_v})")
        write(f"    before freeze (raw phase-1 ranks): {comp_report(raw)}")
        freeze_ranks_at_multiple(raw, _mult, args.skip_layers, v_multiple=_mult_v)
        write(f"    ACHIEVED (this is what the checkpoint has): {comp_report(raw)}")
        fuse_and_prune(raw, args.skip_layers)
        write("    Sigma fused into V and dead directions pruned")
        return raw

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
                            f"K_comp={1-pk/full_k_params:.1%}≥{k_comp_rate:.1%} — "
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
                            f"V_comp={1-pv/full_v_params:.1%}≥{v_comp_rate:.1%} — "
                            f"V alpha frozen (requires_grad=False), V comp loss dropped"
                        )

                    if (k_frozen and v_frozen) or (global_step >= args.alpha_samples):
                        phase2_entered = True
                        reason = (
                            "both budgets reached"
                            if k_frozen and v_frozen
                            else f"alpha_samples={args.alpha_samples} step limit"
                        )
                        # Free phase 1's optimizer before fusing: its Adam moments
                        # are sized for the U/Sigma/V factors fusion replaces, and
                        # holding them through the fuse can OOM.
                        del optimizer, lr_sched
                        gc.collect()
                        torch.cuda.empty_cache()
                        raw_model = enter_phase2(pbar.write)
                        gc.collect()
                        torch.cuda.empty_cache()
                        # Fresh optimizer over the fused model, so phase 1's momentum
                        # does not carry into recovery.
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
                            f"  [Phase 2] step={global_step + 1} ({reason}): KD-only "
                            f"recovery of the fused model for the remaining steps"
                        )

            if (step + 1) % args.log_steps == 0:
                log = {"loss": loss.item(), "kd_loss": kd_loss.item(),
                       "phase": 2 if phase2_entered else 1, "step": global_step}
                if not phase2_entered:
                    pk = meas_k(model)
                    pv = meas_v(model)
                    k_comp = 1.0 - pk / full_k_params
                    v_comp = 1.0 - pv / full_v_params
                    pbar.write(
                        f"  [phase1] step={global_step + 1}"
                        f"  loss={loss.item():.4f}"
                        f"  kd={kd_loss.item():.4f}"
                        f"  comp_k={c_loss_k.item():.4f}  comp_v={c_loss_v.item():.4f}"
                        f"  K_comp={k_comp:.2%}(target={k_comp_rate:.1%})"
                        f"  V_comp={v_comp:.2%}(target={v_comp_rate:.1%})"
                    )
                    _rm = accelerator.unwrap_model(model)
                    pbar.write(f"            now:   {comp_report(_rm)}")
                    if _rounding_on:
                        pbar.write(f"            after round-up it becomes: "
                                   f"{comp_report(_rm, _mult, _mult_v)}")
                    log.update({"comp_loss_k": c_loss_k.item(),
                                "comp_loss_v": c_loss_v.item(),
                                "k_comp": k_comp, "v_comp": v_comp})
                else:
                    # The ranks are fixed now, so there is no compression to track.
                    pbar.write(
                        f"  [phase2] step={global_step + 1}"
                        f"  loss={loss.item():.4f}  kd={kd_loss.item():.4f}"
                    )
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save(accelerator.unwrap_model(model).state_dict(), args.output)
                        saved = True
                        pbar.write(f"  Saved fused checkpoint → {args.output}")
                if wandb_run:
                    wandb_run.log(log)

        print(f"Epoch {epoch + 1} done.")

    if not phase2_entered:
        # --alpha-samples >= --num-samples: the threshold search used every step.
        print("WARNING: training ended inside phase 1, so there was no recovery. "
              "Keep --alpha-samples below --num-samples. Fusing the ranks as they are.")
        del optimizer, lr_sched
        gc.collect()
        torch.cuda.empty_cache()
        enter_phase2(print)
    if not saved:
        torch.save(accelerator.unwrap_model(model).state_dict(), args.output)
    print(f"\nFinal fused checkpoint → {args.output}"
          + (f" (best phase-2 loss {best_loss:.4f})" if saved else ""))

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
