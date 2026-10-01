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

- [x] Add quantization latency tests 
- [ ] Add trained weights file for LongChat, LLaMA-3.1-8B
- [x] Update citation reference
- [ ] Add links for project page, arXiv, PMLR
- [x] Fix kernels for acc analysis
- [x] Pre-release of STAR-KV
- [x] Wire 4-bit KV quantization (`kv_quant.py`) into eval and latency (`--kv-quant`)

## Repository Structure

```
├── model.py                             # Shared: decomposed modules, attention replacement
├── train.py                             # Training script (soft-threshold mechanism)
├── eval.py                              # Evaluation: PPL, zero-shot, LongBench, RULER
├── latency.py                           # Latency benchmarks: layer-wise and end-to-end
├── check_compression.py                 # Report the KV compression encoded in a checkpoint
├── soft_thres_layer.py                  # Learnable soft-threshold function
├── LlamaLoRaAttention_headwise.py       # Low-rank attention module
├── abx_rope_batched.py                  # Triton kernel: fused A@(B@X^T + RoPE) for K, per-head rank
└── kv_quant.py                          # 4-bit KV quantization: format + Triton decode kernels
```

## Installation

1. Clone the repository

```
git clone https://github.com/PriyanshBhatnagar/STAR-KV.git
cd STAR-KV
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
--model lmsys/longchat-7b-v1.5-32k \
--output fused_weights.pt \
--epochs 1 --lr 2e-5 --seq-len 8192 --num-samples 4000 \
--alpha-lr 1e-2 --alpha-samples 3000 --comp-weight-k 0.1 --comp-weight-v 0.1 --kd-weight 1.0 \
--comp-ratio 0.6 --skip-layers 0 1 2 31 \
--rank-multiple-k 16 --rank-multiple-v 32
```

Trains with knowledge distillation from the uncompressed teacher, in two phases over `--num-samples` blocks:

1. **Phase 1:** a learnable soft-threshold truncates the singular values of the K and V projections until the compression budget is met (or `--alpha-samples` blocks have passed).
2. **Phase 2:** each head's K rank is rounded up to a multiple of `--rank-multiple-k` and the V rank to a multiple of `--rank-multiple-v`, Sigma is fused into V, and the remaining blocks recover the fused model with KD alone.

The fused model is written to `--output` — this checkpoint is the only artifact, and is used for both accuracy and latency evaluation.


To check ranks and compression across layers:

```
python check_compression.py --weights fused_weights.pt --per-layer --compressed-only
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

Add `--baseline` (and drop `--weights`) to measure the uncompressed model for comparison. Add `--triton` to evaluate through the fused Triton attention — the same path `latency.py` benchmarks — instead of the pure-PyTorch reference.

### Latency Benchmarks

`latency.py` times the real model: the real weights, attention modules and cache on every decode step. It loads both the uncompressed model and STAR-KV and prints the speedup directly.

#### Layer-wise Latency

```
python latency.py \
  --model lmsys/longchat-7b-v1.5-32k \
  --weights fused_weights.pt \
  --seq 32768 --batch 16
```

Times one decode step of every compressed layer's attention module (`q_proj` through `o_proj`) for both models, and prints each layer's latency, speedup and KV bytes per token, followed by the average over the compressed layers. Only one layer is on the GPU at a time, so 32K × batch 16 fits on a 24 GB GPU; both models stay in host memory (~26 GB).

#### End-to-End Latency

```
python latency.py \
  --model lmsys/longchat-7b-v1.5-32k \
  --weights fused_weights.pt \
  --mode e2e --seq 4096 --batch 1
```

Prefills, then averages 16 decode steps (`--steps`) through the whole model, and prints the time per token, tokens per second and the speedup. The whole model is on the GPU here, so the dense cache has to fit beside the weights; a model that runs out of memory is reported as OOM. At small batch × seq a decode step is bound by reading the weights rather than the cache, so STAR-KV pulls ahead once there are more than ~3–4K cached tokens (batch × seq).

### 4-bit KV Quantization

An add-on to the low-rank cache (`kv_quant.py`). Every K/V latent is quantized per token: each head's leading 20% of channels (the outliers) at 4 bits and the rest at 3 bits, each group with its own scale. Before quantizing, each group is rotated by a Hadamard transform that is folded offline into the low-rank factors, so it adds no work at run time.

Accuracy, with fp32 fake quantization on the PyTorch path (add `--triton` to run through the packed 4-bit cache and kernels instead):

```
python eval.py \
  --model lmsys/longchat-7b-v1.5-32k \
  --weights fused_weights.pt \
  --kv-quant --tasks piqa,openbookqa
```

Latency, with the packed 4-bit cache and the Triton kernels:

```
python latency.py \
  --model lmsys/longchat-7b-v1.5-32k \
  --weights fused_weights.pt \
  --seq 32768 --batch 16 --kv-quant
```

Layer-wise decode latency at batch 16 on an RTX 4090, averaged over the 28 compressed layers of longchat-7b:

| Context | Dense | STAR-KV (bf16) | STAR-KV + 4-bit KV | 4-bit vs dense | 4-bit vs bf16 |
|---|---|---|---|---|---|
| 4K  | 3,727 µs   | 1,306 µs  | 697 µs   | 5.3x | 1.9x |
| 8K  | 7,267 µs   | 2,340 µs  | 1,087 µs | 6.7x | 2.2x |
| 16K | 14,390 µs  | 4,492 µs  | 1,872 µs | 7.7x | 2.4x |
| 32K | 27,860 µs  | 8,912 µs  | 3,474 µs | 8.0x | 2.6x |
| 64K | 54,800 µs* | 17,792 µs | 6,712 µs | 8.2x | 2.7x |

\* A single dense layer does not fit at 64K × batch 16, so this point is extrapolated linearly from 16K and 32K.

The 4-bit decode kernels are MHA-only (e.g. longchat-7b); on GQA models, `eval.py --kv-quant` still measures accuracy through the fake-quant path.


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
