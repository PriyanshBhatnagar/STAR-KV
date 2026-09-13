# [ICML 2026] STAR-KV: Adaptive Low-Rank KV Cache Compression 

Official implementation of the **ICML 2026 Spotlight** paper:

**STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control**

[Paper](#) [Project Page](#) [arXiv](https://arxiv.org/abs/2606.08382) [PMLR](#) <!-- links to be added -->

## Authors

Priyansh Bhatnagar<sup>1\*</sup>, Ashkan Moradifirouzabadi<sup>1\*</sup>, Se-Hyun Yang<sup>2</sup>, SeungJae Lee<sup>2</sup>, Jungwook Choi<sup>3</sup>, Mingu Kang<sup>1</sup>

<sup>1</sup>University of California San Diego &nbsp; <sup>2</sup>Dnotitia &nbsp; <sup>3</sup>Hanyang University


<sup>\*</sup>Equal contribution

## Updates

- [2025.04.30]: 🚀 STAR-KV is accepted at ICML 2026 as a spotlight paper (top 2.2%).

## TL;DR

STAR-KV compresses the KV cache of large language models by caching only low-rank intermediate activations using a learnable soft-threshold to adaptively truncate singular components, and fusing reconstruction, rope and attention via custom Triton kernels to achieve upto 6.9x speed-up.

## Abstract

Low-rank projection is a promising approach for compressing the KV cache because it exploits redundancy along the hidden dimension. However, many prior methods use fixed or heuristic rank selection, which makes it difficult to achieve aggressive compression while maintaining accuracy. We propose STAR-KV, an adaptive low-rank KV-cache compression framework with fine-grained rank control. STAR-KV includes three key techniques. First, it uses a differentiable thresholding mechanism to automatically select the rank at both the attention-head and block levels. Second, it introduces a hybrid decomposition strategy that applies different low-rank factorizations based on the sensitivity of key and value projections. Third, it uses low-rank-aware mixed-precision quantization to leverage data statistics for near-lossless low-bit quantization. Across multiple LLMs and benchmarks, STAR-KV achieves up to 75% KV-cache compression and up to 20x overall KV-cache reduction when combined with quantization. With custom Triton-based GPU kernels, STAR-KV delivers up to 6.9x speedup for the attention module and 3.1x improvement in end-to-end generation throughput.

## Todo Lists

- [ ] Add quantization latency tests 
- [ ] Add trained weights file for LongChat, LLaMA-3.1-8B
- [ ] Update citation reference
- [ ] Add links for project page, arXiv, PMLR
- [x] Fix kernels for acc analysis
- [x] ~~Pre-release of STAR-KV~~
- [ ] Wire the quantized kernels (`bx_quant.py`, `*_quant.py`) into train/eval/latency

## Repository Structure

```
├── model.py                             # Shared: decomposed modules, attention replacement
├── train.py                             # Training script (soft-threshold mechanism)
├── eval.py                              # Evaluation: PPL, zero-shot, LongBench, RULER
├── latency.py                           # Latency benchmarks: end-to-end and layer-wise
├── check_compression.py                 # Report the KV compression encoded in a checkpoint
├── soft_thres_layer.py                  # Learnable soft-threshold function
├── LlamaLoRaAttention_headwise.py       # Low-rank attention module w/ Triton (no quantization)
├── LlamaLoRaAttention_headwise_quant.py # Low-rank attention module w/ Triton (int8/int4 KV cache)
├── abx_rope_batched.py                  # Triton kernel: fused A@(B@X^T + RoPE) for K, per-head rank
└── bx_quant.py                          # Triton kernel: fused dequant + B@X for V
```

## Installation

1. Clone the repository

```
git clone https://github.com/PriyanshBhatnagar/STAR-KV.git
cd StarKV
```

2. Create and activate conda environment

```
conda create -n StarKV python=3.12.7
conda activate StarKV
```

3. Install dependencies

```
pip install -r requirements.txt
```

## Usage

### Training

```
export HF_TOKEN="your_huggingface_token"   # for gated models (e.g. Llama-3)
export WANDB_API_KEY="your_wandb_key"      # optional

python train.py \
--model meta-llama/Llama-3.1-8B-Instruct \
--output fused_weights.pt \
--epochs 1 --lr 2e-5 --seq-len 8192 --num-samples 4000 \
--alpha-lr 1e-2 --alpha-samples 3000 --comp-weight-k 0.1 --comp-weight-v 0.1 --kd-weight 1.0 \
--desired-comp-rate 0.6
```

Trains with knowledge distillation from the uncompressed teacher. The soft-threshold adaptively truncates singular values for both K and V projections. After training, Sigma is fused into V and the result is written to `--output` — this fused checkpoint is the only artifact, and is used for both accuracy and latency evaluation.

Optionally add `--phase3-samples N` for an extra KD-only fine-tune of the already-fused model (overwrites `--output`). Fusion is numerically exact, so this is pure additional recovery rather than a correction.

To confirm the checkpoint actually carries the expected compression:

```
python check_compression.py --weights fused_weights.pt --per-layer
```

### Evaluation

#### Perplexity

To evaluate perplexity on WikiText-2 and C4:

```
python eval.py \
  --model lmsys/longchat-7b-v1.5-32k \
  --weights fused_weights.pt \
  --ppl --ppl-datasets wikitext2,c4
```

#### Zero-shot Accuracy

To run zero-shot evaluations on PIQA, WinoGrande, ARC, HellaSwag, and OpenBookQA:

```
python eval.py \
  --model lmsys/longchat-7b-v1.5-32k \
  --weights fused_weights.pt \
  --tasks piqa,winogrande,arc_easy,arc_challenge,openbookqa,hellaswag \
  --batch-size 32
```

#### Long-Context Benchmarks

To evaluate on LongBench and RULER:

```
python eval.py \
  --model lmsys/longchat-7b-v1.5-32k \
  --weights fused_weights.pt \
  --longbench --ruler --max-length 31500
```

To save all results to JSON, add `--output results/eval_results.json` to any of the above commands.

Add `--baseline` (and drop `--weights`) to measure the uncompressed model for comparison. Add `--triton` to evaluate through the fused Triton attention — the same path `latency.py` benchmarks — instead of the pure-PyTorch reference; note perplexity runs prefill only, so `--triton` matters for the generation-based tasks rather than PPL.

### Latency Benchmarks

#### Layer-wise Latency

```
python latency.py \
  --model lmsys/longchat-7b-v1.5-32k \
  --weights fused_weights.pt \
  --mode layerwise --lw-seq 64000 --lw-batch 16 \
  --lw-cpu-model \
  --output-dir results/
```

Reports per-layer attention latency for BF16 SDPA, low-rank without Triton, and the Triton kernel at both the padded uniform rank and the real per-head rank, using the ranks read from the checkpoint. Results are written to `results/layerwise_latency.csv` — rename it between runs, since each run overwrites the same file.

#### End-to-End Latency

```
# Step 1: baseline (no weights needed)
python latency.py \
  --model lmsys/longchat-7b-v1.5-32k \
  --mode e2e --baseline \
  --output-dir results/

# Step 2: low-rank + Triton
python latency.py \
  --model lmsys/longchat-7b-v1.5-32k \
  --weights fused_weights.pt \
  --mode e2e \
  --ctx-lens 256 512 1024 2048 4096 8192 16384 32000 64000 128000 \
  --output-dir results/
```

If `results/baseline_latency.json` is present when running the low-rank benchmark, a speedup table is printed automatically.


## Reference

If you find this work useful, please consider citing our paper:

```
@misc{starkv2026,
      title={STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control}, 
      author={Priyansh Bhatnagar and Ashkan Moradifirouzabadi and Se-Hyun Yang and SeungJae Lee and Jungwook Choi and Mingu Kang},
      year={2026},
      eprint={2606.08382},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.08382}, 
}
```
