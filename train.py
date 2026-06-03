"""Train a model to achieve adaptive low-rank KV cache compression.

Uses headwise decomposition for K and joint decomposition for V, with learnable
soft-threshold mechanism that finds optimal ranks during training.

Training runs in three phases:
  Phase 1: KD loss + compression loss, alpha LR active.
           Ends when the desired KV compression budget is reached (see
           --desired-comp-rate), with --alpha-samples as a step-count fallback.
  Phase 2 (remaining steps):       KD loss only for recovery.
  Phase 3 (--phase3-samples steps): Sigma fused into V (fixed rank), KD-only minor fine-tune
                                    of fused U/VS modules.  Saved to --output-fused.
                                    Weights load directly into latency.py Triton setup.

Compression budget
------------------
  --desired-comp-rate C  sets the overall KV compression fraction (fraction of params removed).
  C=0.6 means 60% of KV parameters are pruned; only 40% of full-rank capacity remains.
  For the headwise-K / joint-V hybrid, K targets C+0.10 and V targets C-0.10 compression,
  so the average equals C.  Example: C=0.6 → K removes 70% (30% remains), V removes 50%.

Example
-------
  python train.py --model meta-llama/Llama-3.1-8B-Instruct --output trained_weights.pt --output-fused fused_weights.pt --epochs 1 --lr 2e-5 --seq-len 4096 --num-samples 3500 --alpha-lr 1e-2 --alpha-samples 2500 --comp-weight-k 0.1 --comp-weight-v 0.1 --kd-weight 1.0 --desired-comp-rate 0.6 --phase3-samples 200
  python train.py --model meta-llama/Llama-3.2-3B  --output trained_weights.pt --output-fused fused_weights.pt --epochs 1 --lr 2e-5 --seq-len 4096 --num-samples 4000 --alpha-lr 1e-2 --alpha-samples 3000 --comp-weight-k 0.1 --comp-weight-v 0.1 --kd-weight 1.0 --desired-comp-rate 0.6 --phase3-samples 200
  """


import argparse
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
    DecomposeLinear,
    DecomposeLinear_headwise,
    replace_linear_layer,
    collect_K_parameter_size,
    collect_V_parameter_size,
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
# Phase-3 fusion: bake Sigma into V in-place (no Triton, no custom attention)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _fuse_sigma_into_v_inplace(model, skip_layers):
    """Multiply soft-thresholded Sigma diagonal into V.weight for every
    DecomposeLinear / DecomposeLinear_headwise layer, then reset Sigma to
    identity (diag=1, alpha=0) and freeze it.

    After this call the existing U(S(V(x))) forward is numerically U(V_new(x))
    because S is identity.  The model structure stays as-is (LlamaForCausalLM
    with DecomposeLinear modules), so latency.py can still call
    replace_attn_with_triton on the saved weights — export_kproj/vproj_for_triton
    read diag=1, alpha=0 and produce VS = V_new * 1 = V_new, which is correct.
    """
    skip = set(skip_layers)
    for i, block in enumerate(model.model.layers):
        if i in skip:
            continue
        attn = block.self_attn

        # ── v_proj: DecomposeLinear (single global Sigma) ────────────────────
        vp = attn.v_proj
        if isinstance(vp, DecomposeLinear):
            diag = vp.Sigma.diag.detach().float()
            alpha = float(vp.Sigma.soft_thres_layer.alpha)
            s_eff = (diag - alpha).clamp_min_(0)
            vp.V.weight.data = (vp.V.weight.detach().float() * s_eff[:, None]).to(
                vp.V.weight.dtype
            )
            vp.Sigma.diag.data.fill_(1.0)
            vp.Sigma.soft_thres_layer.alpha.data.fill_(0.0)
            vp.Sigma.diag.requires_grad_(False)
            vp.Sigma.soft_thres_layer.alpha.requires_grad_(False)

        # ── k_proj: DecomposeLinear_headwise (per-head Sigma_blocks) ─────────
        kp = attn.k_proj
        if isinstance(kp, DecomposeLinear_headwise):
            if hasattr(kp, "Sigma_blocks"):
                col = 0
                for sb in kp.Sigma_blocks:
                    r_h = sb.diag.numel()
                    diag = sb.diag.detach().float()
                    alpha = float(sb.soft_thres_layer.alpha)
                    s_eff = (diag - alpha).clamp_min_(0)
                    kp.V.weight.data[col:col + r_h] = (
                        kp.V.weight.detach()[col:col + r_h].float() * s_eff[:, None]
                    ).to(kp.V.weight.dtype)
                    sb.diag.data.fill_(1.0)
                    sb.soft_thres_layer.alpha.data.fill_(0.0)
                    sb.diag.requires_grad_(False)
                    sb.soft_thres_layer.alpha.requires_grad_(False)
                    col += r_h
            else:
                diag = kp.Sigma.diag.detach().float()
                alpha = float(kp.Sigma.soft_thres_layer.alpha)
                s_eff = (diag - alpha).clamp_min_(0)
                kp.V.weight.data = (kp.V.weight.detach().float() * s_eff[:, None]).to(
                    kp.V.weight.dtype
                )
                kp.Sigma.diag.data.fill_(1.0)
                kp.Sigma.soft_thres_layer.alpha.data.fill_(0.0)
                kp.Sigma.diag.requires_grad_(False)
                kp.Sigma.soft_thres_layer.alpha.requires_grad_(False)


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
    p.add_argument("--output", default="trained_weights.pt",
                   help="Path to save the best checkpoint")
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
                   help="Overall KV compression fraction (fraction of params removed). "
                        "0.6 means 60%% compressed, so 40%% of full-rank capacity remains. "
                        "K targets rate+0.10, V targets rate-0.10 compression "
                        "(e.g. 0.6 → K removes 70%%, V removes 50%%). "
                        "Phase 2 starts once both targets are met or --alpha-samples is exhausted.")
    p.add_argument("--kd-weight", type=float, default=1.0,
                   help="Weight for the knowledge-distillation KL loss")
    p.add_argument("--comp-weight-k", type=float, default=0.1,
                   help="Weight for the K compression regularisation loss (phase 1 only)")
    p.add_argument("--comp-weight-v", type=float, default=0.1,
                   help="Weight for the V compression regularisation loss (phase 1 only)")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 31],
                   help="Attention layer indices to leave uncompressed")
    p.add_argument("--log-steps", type=int, default=100,
                   help="Print training stats and save checkpoint every N steps")
    p.add_argument("--phase3-samples", type=int, default=200,
                   help="KD-only fine-tune steps after Sigma is fused into V (0 to skip)")
    p.add_argument("--output-fused", default="fused_weights.pt",
                   help="Path to save the best Phase 3 fused checkpoint")
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
    full_k_params = collect_K_parameter_size(model, equal=True)
    full_v_params = collect_V_parameter_size(model, equal=True)
    k_comp_rate = min(args.desired_comp_rate + 0.09, 0.99)
    v_comp_rate = max(args.desired_comp_rate - 0.11, 0.01)
    k_budget = int((1.0 - k_comp_rate) * full_k_params)
    v_budget = int((1.0 - v_comp_rate) * full_v_params)
    print(
        f"Compression targets: overall={args.desired_comp_rate:.0%} removed, "
        f"K={k_comp_rate:.0%} removed ({1-k_comp_rate:.0%} remains, budget={k_budget:,}), "
        f"V={v_comp_rate:.0%} removed ({1-v_comp_rate:.0%} remains, budget={v_budget:,})"
    )

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

                # ── Per-projection budget check; phase 2 when both done ────
                if not phase2_entered:
                    pk = collect_K_parameter_size(model, equal=True)
                    pv = collect_V_parameter_size(model, equal=True)

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
                        # Step-limit fallback: budgets not both met, so freeze any
                        # still-trainable alpha now
                        raw_model = accelerator.unwrap_model(model)
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
                                "weight_decay": 0.01, "lr": args.lr,
                            },
                            {
                                "params": [
                                    p for n, p in raw_model.named_parameters()
                                    if p.requires_grad
                                    and any(nd in n for nd in no_decay)
                                ],
                                "weight_decay": 0.0, "lr": args.lr,
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
                pk = collect_K_parameter_size(model, equal=True)
                pv = collect_V_parameter_size(model, equal=True)
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
                    torch.save(model.state_dict(), args.output)
                    pbar.write(f"  Saved checkpoint → {args.output}")

        print(f"Epoch {epoch + 1} done. Best loss: {best_loss:.4f}")

    print(f"Training complete. Best Phase 1/2 checkpoint saved to: {args.output}")

    # ── Phase 3: fuse Sigma into V, fine-tune fused U/VS with KD only ────────
    if args.phase3_samples > 0:
        print("\nPhase 3: fusing Sigma into V and fine-tuning with KD loss (pure PyTorch, no Triton)...")

        # Fuse Sigma into V in-place; model stays as LlamaForCausalLM with
        # DecomposeLinear modules — no Triton, no custom attention during training.
        raw_model = accelerator.unwrap_model(model)
        _fuse_sigma_into_v_inplace(raw_model, args.skip_layers)

        # New optimizer — as old one references now-gone U/S/V params.
        p3_optimizer = torch.optim.AdamW(
            [p for p in raw_model.parameters() if p.requires_grad],
            lr=args.lr,
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
                torch.save(raw_model.state_dict(), args.output_fused)
                pbar3.write(
                    f"  [phase3] Saved best fused checkpoint → {args.output_fused}"
                )

        print(f"Phase 3 done. Best fused loss: {best_p3_loss:.4f}")
        print(f"Fused checkpoint (U/VS, loadable by latency.py) → {args.output_fused}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
