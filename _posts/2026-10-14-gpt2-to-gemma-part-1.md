---
layout: post
title: "GPT-2 (124M) to Gemma 3-Style (113M) language models, Part 1: Pretraining Lessons from Swapping One Architecture Component at a Time"
date: 2026-10-14
math: true
---

In this article, I will share what I learned from pretraining language models from scratch, and how to build intuition about their inner workings as model architecture and parameter count change. Thanks to epic learning resources available on the internet, I started with Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) project
[*Let's reproduce GPT-2 (124M)*](https://github.com/karpathy/build-nanogpt). I also got inspired by Aleksa Gordić's awesome illustrative article
[*Inside the Transformer: The Life of a Token*](https://www.aleksagordic.com/blog/transformer), where he walks through
Essential AI's Rnj-1.5 7B model, which closely follow the Gemma 3 architecture and uses YaRN to stretch its context
window. It explains grouped-query attention (GQA), rotary position
embeddings (RoPE), YaRN, gated MLPs (GeGLU), RMSNorm and hybrid local/global attention so clearly that I wanted to find out: what do these architecture choices actually buy when you train models at the scale I had started at, and how does data affect what the models can do?

This two-part blog series answers that question. Part 1 (this article) starts from GPT-2 small (124M), called Model A, and changes its architecture one component at a time until it becomes a Gemma 3-style model (113M), called Model B. Both are trained on the same 9.8B tokens of
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), with the same GPT-2 [Byte Pair Encoding (BPE) tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) and AdamW optimizer. [Part 2](https://shinibali.github.io/gpt2-to-gemma-part-2/) looks at how each model
represents position, visualises attention and induction heads, and stretches Model B's context 4x with YaRN. All GPU work (the ablations, the full training runs and the context-extension fine-tuning) ran on rented [RunPod](https://www.runpod.io/) H100s for just under $110. Local debugging, evals and analysis ran on my M1 Pro Mac. Every number in this article comes from code in the [CuriosLM repository](https://github.com/Shinibali/CuriosLM), and every
figure can be regenerated from it. The trained models, with model cards, are on the Hugging Face org [CuriosLM](https://huggingface.co/CuriosLM).

The high-level findings:

- Both Model A and B tie on quality: Validation loss ends at 3.055 vs 3.051, and no benchmark difference is statistically
  significant. The Gemma recipe B reaches a lower validation loss early in training, but by 10B tokens GPT-2 (A) matches up.
- Only RoPE improves quality at this model size: Going from Model A to B one component at a time, grouped-query attention, GeGLU with sandwich RMSNorms
  and local attention don't move the loss beyond seed noise (the difference between two runs that differ only in their random seed).
- Model B needs fewer FLOPs but trains slower: It needs 9% fewer FLOPs per token but trains 4%
  slower, because element-wise operations like the extra norms, gating and rotation are memory-bound work that FLOP counts don't see. Model FLOPs utilization (MFU) drops from 37% (Model A) to 32% (Model B).
- Model B is cheaper for inference: Its grouped-query attention and local windows make its KV cache 6.4x smaller, so it can serve
  longer contexts and more sequences compared to Model A. At this model scale, whether that makes token generation faster
  depends on the hardware (Part 2).
- Data matters more than architecture: Against the original GPT-2 [[1]](#ref-1), which was trained on different data, both models gain about 8 accuracy points on science questions (ARC-Easy benchmark) and lose 8 on predicting the last word of a story passage (LAMBADA benchmark). I traced most of that loss to one capability: copying a character's name from earlier in the story.
- Changing the data gives back part of that capability: Swapping half the training data for synthetic
  textbooks and stories (Cosmopedia-v2) raised LAMBADA accuracy 2.3 points, at
  the cost of 12.7% higher perplexity on FineWeb-Edu.

The specific numbers may not hold at larger scale, but the measurement methods should.

## Setup:

### Model architecture

Model A is GPT-2 small [[1]](#ref-1), exactly as [nanoGPT](https://github.com/karpathy/nanoGPT) builds it. It has 12 layers ($$L$$), 12 attention heads, a model dimension
($$d_{model}$$, or simply $$d$$) of 768, and a 1024-token context, with
learned absolute position embeddings, full multi-head attention in every layer, a GELU MLP, and LayerNorm before each
sub-layer.

Model B keeps the same skeleton (12 layers, 12 query heads, $$d = 768$$, 1024 tokens) and swaps in Gemma 3-style
components. It is **not an exact copy of Gemma 3**: I only took the architectural changes from Gemma-3, while keeping the parameter count upper bounded to GPT-2 small.

What each change is meant to do, what it did in these runs, and where to read more (the inference column is measured
in Part 2):

| Change | What it does | During training | At inference | Deep-dive references |
|---|---|---|---|---|
| RoPE | instead of learned positions, queries and keys are rotated by an angle so attention sees *relative* distance | The only change that improved loss (−0.085 at 1.5B tokens); 0.8M fewer parameters | No effect on cache size; makes context extension (YaRN) possible | Su et al. 2021 [[3]](#ref-3); Gordić's [article](https://www.aleksagordic.com/blog/transformer) |
| Grouped-query attention | 12 query heads share 3 key/value heads (4:1 Q-KV head ratio) | No measurable quality change; 8% fewer FLOPs per token; the fastest of the ablation runs below | 4× smaller KV cache | Ainslie et al. 2023 [[4]](#ref-4) |
| GeGLU + sandwich RMSNorm | GeGLU is a gated variant of the GELU MLP. A gated MLP has three weight matrices instead of two, so its inner width is cut to 2/3 of GPT-2's (2048 instead of $$4d$$ = 3072) to keep the parameter count level. "Sandwich" norm applies RMSNorm both before and after each attention and MLP block, for smoother optimization | Within noise; same FLOPs (narrower MLP); slower, from extra elementwise operations | No effect | Shazeer 2020 [[5]](#ref-5); Zhang & Sennrich 2019 [[6]](#ref-6) |
| Local/global attention | 3 of every 4 layers only attend to the most recent 512 tokens; every fourth layer sees the full context. Per layer, attention looks like *GLLLGLLLGLLL* | Within noise at a 1024-token context; slower unless the kernel skips masked blocks | In 3 of 4 layers, the cache stops growing at 511 past positions | Gemma 2 and 3 reports [[7]](#ref-7), [[8]](#ref-8) |

Gemma 3's local window is 1024 tokens, which would cover
this model's entire 1024-token context and make its local layers global, so Model B's window is half the
context, 512 tokens. Model B also keeps one global layer in every four rather than Gemma 3's one in six, and leaves
out Gemma 3's QK-norm, which I didn't test.

| | Model A (GPT-2 recipe) | Model B (Gemma 3 recipe) |
|---|---|---|
| Parameters | 124.5M | 113.1M |
| Position information | learned table, 1024 × 768 | RoPE (no parameters) |
| Attention | 12 heads, all global | 12 query / 3 KV heads; 3 of 4 layers local (512) |
| MLP | GELU, 3072 wide | GeGLU, 2,048 wide |
| Norm | LayerNorm, pre | RMSNorm, pre and post |

![The two architectures side by side: GPT-2 and the Gemma 3-style recipe, as trained](/assets/s1/figures/architecture.png)

Model B ends up with 9% fewer parameters, and almost all of the difference is GQA's smaller key/value projections. I
didn't enlarge $$d$$ to scale up the parameter count, because the comparison holds the shape of the transformer blocks fixed, and does not fixate the parameter total. At this size, a large share of the parameters sits in the token
embedding table: 31% of Model A's. Google's smallest Gemma, [Gemma 3 270M](https://developers.googleblog.com/en/introducing-gemma-3-270m/) is more extreme, about 170M of its parameters are in the
embedding table, because its vocabulary has 256K entries, and only about 100M are in its transformer blocks. Our models
use GPT-2's BPE tokenizer's 50K vocabulary, so their embedding table is far smaller, and Model B's transformer blocks (about 74M
parameters) are the same order of magnitude as Gemma 3 270M.

The embedding share is itself a small-model effect. With a vocabulary of $$V$$ entries, the embedding table holds $$V·d$$ parameters, while the $$L$$ transformer blocks hold $$\sim 12·L·d^2$$: $$4d^2$$ in each
layer's attention projections (Q, K, V and output) and $$8d^2$$ in its MLP (up and down projections). So the table's share falls quickly with size: 31% of GPT-2 small, 15% of GPT-2 medium, 5%
of GPT-2 XL. At 124M, whether you account for the embedding table changes the parameter count a lot but at larger scale, it barely does.


### Data and tokenizer

Following nanoGPT's recipe, both models train on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (the `sample-10BT`
subset). Hugging Face built it by scoring web pages for educational value; in their
ablations it was the strongest open web corpus for small models. Both our models use GPT-2's BPE tokenizer (vocabulary $$50257$$). It is byte-level, so any text (emojis included as we will see in Part 2) can be encoded without
"unknown" tokens. As in nanoGPT, the embedding and output layers pad the vocabulary from 50257 up to $$V = 50304$$, the next multiple of 64. Why multiple of 64? GPUs compute matrix products in fixed-size tiles, so a dimension that doesn't divide evenly leaves a last tile that is mostly empty work. For 16-bit math, NVIDIA recommends dimensions that are multiples of 64 [[26]](#ref-26). Besides, The 47 extra vocabulary rows we appended are never used as target during training, so the model learns to assign near-zero probability to them.

The data is tokenized once, into 100M-token shards (99 training and 1 validation). Each document is
prefixed with GPT-2's end-of-text token, `<|endoftext|>` (EOT), and the documents are concatenated back to back:

```
[EOT, doc 1 tokens…, EOT, doc 2 tokens…, EOT, doc 3 tokens…, …]
```

A shard is cut every 100M tokens, so a document that runs past the cut continues at the start of the next shard. The
EOT marks where one web page ends and the next begins, so the model learns not to read one into the other. Training
sequences are cut from this stream every 1024 tokens, regardless of document boundaries.


### Training

Both models use nanoGPT's GPT-2 recipe unchanged: AdamW optimizer with a peak learning rate of 6e-4 decaying to 6e-5 on a
cosine schedule, $$\beta_2= 0.95$$, weight decay $$0.1$$ and gradient clipping at $$1.0$$. Training runs in bf16 with
`torch.compile` on rented H100s, with PyTorch's Distributed Data Parallel (DDP): every GPU holds a full copy of the
model, processes its own slice of the data, and the GPUs average their gradients before each optimizer step.

Three numbers dictate how much data a step sees. Each GPU processes a micro-batch of $$B = 60$$ sequences of $$T = 1024$$
tokens at a time. Gradients from $$G = 8$$ micro-batches are accumulated before the optimizer updates the weights, and $$G$$ is split across the GPUs (4 micro-batches each on 2 GPUs, 2 each on 4). So every optimizer step sees $$B·T·G = 491520$$ tokens, however many GPUs share the work.

I didn't tune that step size. GPT-3 [[2]](#ref-2) trained its 125M model with batches of about 0.5M tokens, and
nanoGPT's GPT-2 config reaches ~0.5M as 491,520 tokens (12 sequences × 1024 tokens × 40 micro-batches). Keeping that
number keeps the optimization identical to nanoGPT; only the split between $$B$$ and $$G$$ was chosen for the hardware, with $$B·G = 480$$
sequences fixed.

Both models read the shards in the same order. On each micro-batch, the $$N_{GPU}$$ GPUs together read one contiguous
block of $$N_{GPU}·B·T$$ tokens from the current shard, GPU $$r$$ reading the $$r$$-th slice of $$B·T$$ tokens, and then all
of them stride forward by $$N_{GPU}·B·T$$ tokens. Over one optimizer step they cover 491520 consecutive tokens.



## The ladder: one change at a time

### Step-wise changes at small scale
Architectural changes (and also hyperparameter sweeps) are cheapest to test on short runs, so the project has two stages. First, a ladder of five runs (let's call them "rungs") of 1.5B tokens each (~15% of the full budget), each rung adding one change to the one before; each took about 0.5 hours on 2 H100s. Then the two ends of the ladder, GPT-2 small and the full Gemma 3-style recipe, trained on the full 9.8B tokens (about 1.6 hours each on 4 H100s), with the same
data and the same hyperparameters. Each run's settings came from one model config and a couple of environment variables, and every run logged
to my Weights & Biases account.

| Rung | Adds | Parameters | Val loss | Step vs previous |
|---|---|---|---|---|
| a0 | nothing: GPT-2 as nanoGPT ships it | 124.5M | 3.495 | |
| a1 | RoPE instead of learned positions | 123.7M | 3.410 | **−0.085** |
| a2 | grouped-query attention, 4 query heads per K/V head | 113.1M | 3.431 | +0.021 |
| a3 | GeGLU MLP, RMSNorm before *and* after each sub-layer | 113.1M | 3.454 | +0.023 |
| a4 | local attention (512-token window) in 3 of every 4 layers | 113.1M | 3.455 | +0.001 |

Each step is the difference between two single training runs, so it carries the run-to-run noise of both. To size
that noise, I trained one rung (a4) twice with different random seeds. The two copies ended 0.020 apart; the figure draws that gap as the noise band.

![Each rung's step against the seed noise band](/assets/s1/ladder/figures/ladder_loss.png)

RoPE's step is four times the seed-to-seed gap. It is the one change that makes the model
better at this scale, and it does so with *fewer* parameters, since the learned position table (0.8M parameters)
disappears. GQA and GeGLU each cost about 0.02 in val loss, right at the edge of the band. Either could be real or noise,
and it would take two or three more seeds per rung to conclude. I didn't run those extra seeds, since the GPU hours were better spent on the full-length runs. Local attention changes nothing measurable, as expected at a 1024-token context, where a local
512-token window still sees half of all previous tokens in the full context.

### Training throughput and KV cache per rung

![Training throughput and KV-cache size per rung](/assets/s1/ladder/figures/ladder_efficiency.png)

The right panel shows how GQA and local attention helps at inference time. When a model generates text, it stores the keys and values of every past token (the KV cache). GQA cuts that cache by 4x, from 75 to 19 MB per 1024-token sequence, because 12 query heads now share 3 key/value heads. Local windows take it down to 12 MB (6.4x smaller than Model A), because three layers in four keep only the last 511 positions (the 512-token window includes the current token). These sizes assume float32 keys and values: for Model A, 2 (K and V) × 12 layers × 1024 tokens × 768 dimensions × 4 bytes $$\approx$$ 75 MB. In bf16, the size halves.

The left panel shows that the Gemma 3 recipe trains *slower* at 820k tokens per second against GPT-2's 940k.
Two rungs each cost about 8.5% extra time. One is RoPE, whose rotations add elementwise work on every query and key. The other is local
attention. Restricting attention to a window meant handing PyTorch's scaled dot-product attention function an
explicit mask, and PyTorch can't run its FlashAttention kernel [[14]](#ref-14) with an arbitrary mask. It falls back to a
general-purpose kernel that scores every query-key pair, including the ones the window then throws away.
FlexAttention [[15]](#ref-15) builds the window into a block mask instead, and skips the tiles of the attention matrix that
are fully masked. On the same two H100s, it cut rung a4's time per optimizer step from about 590 ms to about 560 ms
(about 5% faster, MFU 33.4% → 35.2%), but didn't close the whole gap. The ladder figure above shows the dense-mask
version but the final 9.8B-token Model B run used FlexAttention.

GQA, the one change that cuts training FLOPs by a meaningful amount (8%; the local windows cut another 1%), is the
fastest rung. In short, at 124M, RoPE is the change that helps quality, GQA and local windows reduce serving cost, and
GeGLU with sandwich norm can't be told apart from noise. A general intuition about "why sandwich norms?" from literature survey [[27]](#ref-27): Normalizing each sublayer's output as well as its input caps how much any one layer can add to the residual stream, which keeps activations and gradients in a steady range as the model trains .

### How this compares with published results

The papers behind these changes, at billions of parameters, report the same split between quality and cost:

- The GQA paper (Ainslie et al. 2023) [[4]](#ref-4) finds GQA close to full multi-head attention quality,
  at close to multi-query attention speed.
- Gemma 3's report [[8]](#ref-8) describes its move to more local layers and shorter windows as cutting KV-cache memory with little
  effect on perplexity.
- RoPE has become the standard position encoding in open models, including Llama, Qwen and Gemma.
- GLU variants are where the ladder and the literature differ: Shazeer (2020) [[5]](#ref-5) reports lower perplexity from gated MLPs
  in T5 pretraining, while here the change sits at the edge of seed noise.

## Confirming at our full 9.8B token budget

A change that helps at 1.5B tokens can flatten out once the model has seen more data, so I trained the two ends of the ladder on the full budget: Model A (vanilla GPT-2) and Model B (the full Gemma 3-style recipe). Each run is 20,000 steps of 491,520 tokens, or 9.83 billion FineWeb-Edu tokens: a round number of steps that stays just inside the 99 training shards (9.9B tokens), so it is a single pass with no
repeated data. That's roughly 4x the "compute-optimal" budget the Chinchilla scaling law [[9]](#ref-9) suggests for this model size.

The loss throughout is cross-entropy: the average negative natural log of the probability the model gave each correct next
token, over $$N$$ tokens. Because it uses the natural log, its unit is the *nat*. Perplexity is the same quantity exponentiated, and can be interpreted as the number of tokens the model is effectively choosing between:

$$\text{loss} = -\frac{1}{N}\sum_{i=1}^{N} \ln p(x_i \mid x_{<i}) \qquad \text{perplexity} = e^{\text{loss}}$$

![Validation loss of both models over 9.8B tokens, and the gap between them](/assets/s1/headline_comparison/figures/loss_overlay.png)

Both curves start near $$\ln (50257) ≈ 10.8$$, the loss of guessing uniformly over
50257 tokens (excluding the padded vocabulary). Train and validation loss track each other the whole way, as expected when no example is seen twice. The first 0.5B tokens cut the loss by 2.75x; after that it improves slowly. The bottom panel shows the gap between the two models' validation loss.

Model B pulls ahead immediately, which our ladder rungs attributes mostly to RoPE. After 0.5B tokens, B is ahead by 0.196, ten times the
seed-noise gap. Then the gap closes. By about 4.4B tokens it is inside the seed-noise band, and at the
end it is 0.005, with B at 3.0506 and A at 3.0553. That's a tie as the gap is a quarter of the difference between two
seeds of the same model. At this size, Model B's architecture gets to a given loss sooner but does not end lower. Whether the two endpoints separate at
larger scale is a question these runs can't answer.


## Training on a budget: FLOPs, MFU and what a run costs

### Counting FLOPs

The counts below are for the forward pass: what it costs to run one token through the model, which is also what
inference pays per token. Training adds a backward pass, which costs about twice the forward, so about 3x in total.

Almost all of a transformer's arithmetic is matrix multiplication. Multiplying a token's $$m$$-dimensional vector by an $$m × n$$ weight matrix costs ~$$2mn$$ FLOPs: one multiply and one add per weight. Applying this to each piece of a
layer gives **266M** FLOPs per token for Model A and **241.2M** for Model B; the full
derivation is in the appendix ([Counting the FLOPs](#counting-the-flops)).

The count leaves out everything that isn't a matrix multiplication: biases, normalization, activations, softmax and
masking, RoPE's rotations, residual additions, the embedding lookup and the loss. By a rough count they add up to under 1% of
the FLOPs, but they take far more than 1% of the time: each one reads and writes whole tensors from memory while
doing very little arithmetic, as [How well the GPUs were used](#how-well-the-gpus-were-used) shows below.

With $$d$$ being the
model dimension, $$T$$ the context length and $$k̄$$ the average number of keys a query attends to, four insights to notice:

- GeGLU costs exactly what GELU costs: With the hidden width trimmed to $$8d/3$$, both MLPs land at $$16d^2$$ FLOPs, so any
  difference between rungs a2 and a3 can't be a compute difference.
- GQA's saving is in the Q, K, V projections: which drop from $$6d^2$$ (A) to $$3d^2$$ (B) per layer. That is 21.2M FLOPs per token, 8 of the 9 percentage points
  Model B saves over Model A. The local windows supply the last point (3.5M) by shortening $$k̄$$ in nine layers.
- Attention is the only term that grows with context length: Per layer, the attention scores cost $$2d(T + 1)$$
  against $$24d^2$$ for the weight matrices, a ratio of $$(T + 1)/(12d)$$. At $$T = 1024$$ and $$d = 768$$ that's 11%, so
  attention is about $$10\%$$ of each layer's FLOPs (about $$7\%$$ of the whole forward pass, once the output layer is
  counted). At this size and context, local windows don't save many FLOPs; their payoff is in the KV cache.
- Without the attention term, the forward pass is 2 FLOPs per weight: and
  training (the forward pass plus a backward pass costing about 2x the forward) is $$2 + 4 = 6$$ FLOPs per weight:
  the well-known $$6N$$. The attention term adds $$6·L·d·(T + 1)$$ on top.

### Where the FLOPs go

![Forward FLOPs per token, by component](/assets/s1/efficiency/figures/flops.png)

At this model scale, the output layer is almost 1/3 of the total compute: projecting onto the
50,304-entry padded vocabulary costs 2/3 as much as the whole MLP stack.
Training costs about 3x the forward pass: 798 MFLOP per token for Model A and 724 for Model B. Over 9.83B tokens, that's $$7.8 × 10^{18}$$ and $$7.1 × 10^{18}$$ FLOPs. The plain $$6N$$ rule comes within 7% of these numbers; the
difference is the attention term.

### How well the GPUs were used

Model FLOPs utilization (MFU) is the FLOPs a run actually achieves per second, divided by what the hardware can do
at peak. For an H100 SXM in bf16, the peak is 989 TFLOP/s: [NVIDIA's spec sheet](https://www.nvidia.com/en-us/data-center/h100/)
lists 1,979 TFLOP/s "with sparsity", and dense matrix multiplication gets half that.

| Headline runs, 4× H100 | Model A | Model B |
|---|---|---|
| Throughput | 1.82M tokens/s | 1.76M tokens/s |
| MFU, as the training script logs it (PaLM formula) | 39.4% | 35.1% |
| MFU, counting only useful FLOPs | 36.8% | 32.1% |
| GPU-hours for 9.83B tokens | 6.0 | 6.2 |

The training script, following nanoGPT, logs MFU with the PaLM paper's [[11]](#ref-11) formula. It counts $$12·L·d·T$$ FLOPs for attention, which charges every token for attending to
the whole context. Counting only the attention a causal model actually computes (and, for Model B's local layers, only the keys inside the window) gives the "useful FLOPs" row, a few points
lower. B loses more, because the PaLM formula also charges its local
layers for keys they skip. The gap is small at 1024 tokens and grows with context; [Part 2](https://shinibali.github.io/gpt2-to-gemma-part-2/) works out by how much.

![MFU across the two 9.8B-token runs, as logged and counting useful FLOPs only](/assets/s1/efficiency/figures/mfu.png)

The training script logs a running average of MFU, in which each new step's reading counts for 10%, and it leaves
out the steps that also ran an evaluation or saved a checkpoint. So the remaining dips are real stretches where steps
ran slower, for reasons the logs don't record. Model B does 9% fewer FLOPs per token, yet it runs 4% slower, so it spends more time on work that isn't matrix multiplication:
- two extra normalizations per block;
- the gating multiply in GeGLU;
- the rotations in RoPE;
- the attention kernel's handling of the local windows.

Each of these is cheap in FLOPs but has to read and write whole activation tensors from memory. In [Horace He's terms](https://horace.io/brrr_intro.html),
they're memory-bandwidth-bound, not compute-bound, which is why fewer FLOPs did not mean a faster run.

### How the balance shifts as models and contexts grow

The equations in [Counting the FLOPs](#counting-the-flops) show how each component's share of the FLOPs moves with scale. The MLP and attention
projections grow with $$d^2·L$$, the output layer with $$d·V$$, and the attention scores
with $$d·T·L$$, the only term that depends on context. The figure below holds the context at 1024 tokens and follows the GPT-2 recipe from 124M up through GPT-2 XL (1.6B) to a 6.7B model: the output layer shrinks from 29% of the forward pass to 5% and then about 3%, and the MLP becomes the majority.

![Each component's share of a model's forward FLOPs, from 124M to 6.7B parameters](/assets/s1/efficiency/figures/flops_by_size.png)

Two crossovers follow from the same terms. Attention overtakes the MLP at $$T ≈ 8d$$, about 6,000 tokens at our $$d$$ of 768; [Part 2](https://shinibali.github.io/gpt2-to-gemma-part-2/) follows that term to long contexts. The crossover that matters at this
size is the output layer: its $$2dV$$ FLOPs outweigh all the MLPs together ($$16d^2·L$$) while $$d·L < V/8$$. With the padded 50304-entry
vocabulary, $$V/8 ≈ 6288$$. Our models sit at $$d·L = 9216$$, just past the crossover, which is why the output layer costs
2/3 as much as the MLPs here. A model with Gemma 3's 262,144-entry vocabulary needs $$d·L$$ above about 33,000 before
its MLPs outweigh its output layer.

### Getting the GPUs busy in the first place

The first runs on the H100s reached an MFU of only about 25%, with a micro-batch $$B$$ of 12 sequences per GPU. A
124M model is small enough that, at 12 sequences per GPU, a large share of every step goes to fixed
overheads: launching kernels, moving small tensors, synchronizing. Raising the micro-batch to 60 sequences, and
reducing gradient accumulation to keep each optimizer step at 491,520 tokens, took MFU up to 41%. The optimizer update is mathematically identical, since
every step still sees the same tokens. Beyond 60 sequences the gain flattened out while
memory kept climbing. And because the $$G$$ accumulated micro-batches are split across the GPUs, moving from 2 to 4 GPUs sped training up 1.94x with identical optimizer steps.

The training data never sits in GPU memory. The 100 tokenized shards (100M tokens × 2 bytes each) take 20 GB of disk.
They live in our [CuriosLM Hugging Face org](https://huggingface.co/CuriosLM), each pod downloads them in about a minute, and the data loader keeps only the current
200 MB shard in CPU memory. For comparison, a 15T-token corpus is 30–60 TB
once tokenized (2 to 4 bytes per token) which is too much to copy onto every machine, so larger runs typically streams training tokens from shared cloud storage.

### How large a model fits on one GPU

Each GPU's memory (80 GB of HBM on an H100) holds two main things: the **model state**, which grows with the parameter count $$N ≈ 12·L·d^2 + V·d$$, and the **activations** saved for the backward pass, which grow with the micro-batch of $$B$$
sequences of $$T$$ tokens. The table gives the standard estimate for each, next to Model A's measured peak memory:

| Per GPU | Bytes | Model A, $$B = 60$$, $$T = 1024$$ |
|---|---|---|
| Model state: bf16 weights and gradients, float32 master weights, AdamW's two moments [[10]](#ref-10) | $$16N$$ | 2.0 GB |
| Activations in the transformer blocks [[25]](#ref-25) (without the $$T × T$$ attention scores, which FlashAttention never stores) | $$34·B·T·L·d$$ | 19.3 GB |
| Activations of the output layer: float32 logits for the loss (4 bytes each [[25]](#ref-25)) plus the bf16 logits the layer outputs (2 bytes) | $$6·B·T·V$$ | 18.5 GB |
| Estimated | | 39.8 GB |
| Measured peak | | 41.4 GB |

At this size, the output layer's activations are as large as all twelve transformer blocks', the same small-model effect as in
the FLOPs. Since the blocks hold $$N ≈ 12·L·d^2$$
parameters, their activations $$34·B·T·L·d$$ equal $$17N · B·T/(6d)$$. Rounding the 17 down to 16 (a 6% error) puts both terms in the same form:

$$\text{training memory } M ≈ 16N\left(1 + \frac{B·T}{6d}\right) + 6·B·T·V $$

The activations equal the model state once a micro-batch holds about $$6d$$
tokens, so small models
are limited by activations and large ones by their state. The activations are the flexible part: a smaller micro-batch $$B$$, made up with more gradient accumulation,
keeps the same 491,520-token optimizer step, and activation checkpointing recomputes activations instead of storing
them. The model state can't shrink, so it sets the ceiling. Solved for $$N$$, the equation gives the largest model a GPU with memory $$M$$ can train. With the smallest micro-batch, $$B = 1$$ sequence of $$T = 1024$$ tokens, and a width typical of 3–5B-parameter models ($$d ≈ 3000$$), $$B·T/(6d) ≈ 0.06$$ and
the output layer's $$6·B·T·V$$ is only 0.3 GB, so for an 80 GB H100:

$$N_{max} ≈ \frac{M - 6·B·T·V}{16\left(1 + \frac{B·T}{6d}\right)} ≈ \frac{80\text{ GB} - 0.3\text{ GB}}{16 × 1.06} ≈ 4.7\text{B}$$

Leaving a few GB for the CUDA context and workspace, which the equation doesn't
count, brings it to about 4.5B. Larger models have to be split across GPUs: ZeRO [[10]](#ref-10) (or PyTorch's FSDP) shards the optimizer state, gradients and weights across the data-parallel GPUs; tensor parallelism splits each layer's weight matrices across GPUs (Megatron-LM [[12]](#ref-12)); and pipeline parallelism places different layers on different GPUs.



## Interlude: the step-320 spike

While monitoring the first ladder runs, I saw a sharp spike in training loss at step 320. Loss on a single batch is noisy and spikes happen, but this one appeared at the same step in every run: both
architectures, short runs and long ones, on 2 GPUs and on 4. The only thing these runs share is the order of the
training data, so the likely cause was the data itself.

![The training loss of three different runs around step 320](/assets/s1/step320_spike/figures/loss_spike.png)

Because the data loader is deterministic, the batch can be replayed exactly. I rebuilt the data loader with the run's
settings, stepped it forward to the micro-batch logged at step 320, and scored each of its 60 sequences. One
sequence stands far above the rest, with a mean loss of 5.06 against a batch median of 3.83.

It's an encyclopedia-style educational page, but with
some words rewritten by a ["baby talk" text converter](https://stackoverflow.com/beta/challenges/79640866/challenge-1-implement-a-text-to-baby-talk-translator-completed): *"a totaw of", "bewow de age of 18 years", "broders and cwose rewatives",
"de medod used".*

The converter swaps *l* for *w* and *th* for *d* (and, more rarely, a word-final *th* for *f* and *qu* for *qw*). For the model it's close to a
different language, and the tokenizer is where it breaks. `close` is one token, but `cwose` is
three (`c·w·ose`), each of them rare in that position. Other words turn into different words: `the` becomes ` de`, a
perfectly common token that means something else entirely, so the model has to predict a real word in the wrong place.

![Loss on each mangled word, real spelling vs baby talk](/assets/s1/step320_spike/figures/word_losses.png)

Comparing the page with a cleaned-up copy shows where the extra loss lives. In the excerpt shown in the figure, 11 mangled words
account for 135 of its 165 extra nats. The other 30 spill onto the words after them: once the model has seen
"bewow de", even an ordinary "age" becomes harder to predict.

The cleaned copy came from a model trained in this study: the 1.5B-token checkpoint of ladder rung a4. For every word
containing a *w*, *d*, *qw* or word-final *f*, I generated every spelling the converter's rules could have mangled
into it (each *w* may be a real *w* or an *l*, each *d* a real *d* or a *th*, and so on), and let the checkpoint pick
the one it finds most likely after the already-corrected text before it. That fixed 196 words
across the page, with one miss: a foreign word with a genuine *d* came out with a *th*, because the model had no way
to know the *d* was real.

Pages like this are in every web corpus, and this one did no lasting damage. The loss is back to normal on the next step, because the page is one of 60 sequences in the logged micro-batch (which is why the spike is visible) but
only one of 480 in the optimizer step. In GPT-3's [[2]](#ref-2) largest batch of about 3.2M tokens, it would be one in about 3,000.
Spikes in large runs therefore tend to come from the optimization itself rather than from a single bad document, and
they are harder to recover from. The PaLM paper [[11]](#ref-11) reports about 20 spikes while training its 540B model. Its fix was to
restart from a checkpoint roughly 100 steps before each spike and skip the 200–500 data batches around it. That only
works if you can pinpoint exactly which batches those were, which is what a deterministic data loader provides, and it is also what made this diagnosis quick.

## Evaluation: a tie on every benchmark

Validation loss says the two models are the same. But loss is an average over every token, and two models can reach the same
average by being good at different things. So I evaluated both on a set of standard benchmarks, with OpenAI's original GPT-2 (124M) checkpoint [[1]](#ref-1)
([`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) on Hugging Face) alongside as a reference, called "stock GPT-2" from
here on. All three go through the same harness, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
connected to the models through a small custom adapter.

### Choosing benchmarks for a 124M model

A benchmark is only useful if the model scores clearly above chance and below the ceiling. Otherwise every
model lands on the same number and the comparison says nothing. Of the benchmarks commonly used for small models, four measures give a usable signal at this size:

- HellaSwag [[19]](#ref-19): pick the sensible ending to an everyday scenario from 4 options. It's the standard yardstick for small models: GPT-2-sized models score around 30%, against 25% for random guessing. Each ending is scored by its average log-probability per character, so a long ending isn't penalized just for being long (lm-eval's `acc_norm`).
- ARC-Easy [[20]](#ref-20): grade-school science questions. These are knowledge-heavy, so this is where training data should show.
- LAMBADA [[21]](#ref-21): predict the last word of a passage, where the word can only be guessed from the wider passage. It is the closest of the four to a test of long-range context, which is where the two models' position schemes differ.
- Perplexity on held-out text from FineWeb-Edu (the training distribution) and Cosmopedia-v2 (synthetic textbooks). With no question-answer format, it tests how well each model predicts real text, and it is the least noisy number available.

MMLU is left out because models this size sit at chance (25%), so it can't tell two of them apart.

### Results

Stock GPT-2 was trained on different data (WebText), so it is here for reference.

|Accuracy in % | HellaSwag  | ARC-Easy  | LAMBADA  |
|---|---|---|---|
| Model A (GPT-2 recipe) | 32.4 | **52.4** | 24.2 |
| Model B (Gemma recipe) | **32.5** | 51.9 | 24.5 |
| A vs B: p (paired)¹ | 0.84 | 0.63 | 0.57 |
| *Stock GPT-2 124M* | *31.1* | *43.8* | ***32.5*** |

| Held-out perplexity  | FineWeb-Edu | Cosmopedia-v2 |
|---|---|---|
| Model A | 21.89 | 20.31 |
| Model B | 21.80 | 20.49 |
| B − A, 95% CI¹ | [−0.12, −0.06] | [+0.15, +0.22] |
| *Stock GPT-2 124M* | *26.40* | *20.54* |

¹(A vs B is paired, since both models answer the same questions. Accuracies use McNemar's test on per-question
correctness (McNemar, 1947) [[23]](#ref-23). Perplexities use a bootstrap over documents (Efron, 1979) [[24]](#ref-24), 10,000 resamples, because
tokens within a document aren't independent. Both follow [Miller (2024), *Adding Error Bars to Evals*](https://arxiv.org/abs/2411.00640) [[22]](#ref-22).
Run-to-run (seed) noise is measured separately, from two seeds of one ladder rung.)

**None of the benchmark gaps between A and B is significant.** The p-value in the table answers one question: if the
two models were equally good, how often would a gap at least this large show up just from which questions happen to be
in the test? On HellaSwag, A and B gave different answers on 841 of the 10,042 questions; A got 417 of those right and
B got 424. If each of those disagreements were a fair coin flip, a split at least this uneven would happen 84% of the
time (p = 0.84), so chance explains it easily. The usual bar for calling a difference real is p < 0.05.

On perplexity, B is 0.09 better on FineWeb-Edu and 0.18 worse on Cosmopedia-v2. The 95% confidence interval (CI) is
the range the B − A gap would likely fall in if the test documents were resampled, and both intervals exclude zero. That makes both differences real *for these two checkpoints*, but only in the
sense that a different sample of test documents wouldn't change them. Retraining either model with a different
random seed moves its loss by about 0.02, roughly 0.4 perplexity, which is two to four times larger than either
gap. Each checkpoint wins on one set and loses on the other.

The stock GPT-2 row also checks the harness: an independent lm-eval run of the same checkpoint,
published on [AMD's model card for AMD-Llama-135m](https://huggingface.co/amd/AMD-Llama-135m), matches it to within
0.001 on ARC-Easy, HellaSwag and LAMBADA. Against that row, both of our models are 8 points better on ARC-Easy and 8 points worse on LAMBADA.


## Data shapes capability more than architecture

Changing the architecture did not move any benchmark significantly. Stock GPT-2 shows what changing the data does: it has the
same size and architecture as Model A but different training data. Stock GPT-2 was trained on WebText, pages linked from Reddit (2019); Models A and B
on FineWeb-Edu, web pages scored for educational value (2024).

Against GPT-2, both models gain on some tasks and lose on others, by large margins and in the same direction:

| vs stock GPT-2 | Model A | Model B |
|---|---|---|
| ARC-Easy (grade-school science) | **+8.6 pt** | +8.1 pt |
| HellaSwag (everyday scenarios) | +1.3 pt | +1.3 pt |
| LAMBADA accuracy (last word of a story) | **−8.2 pt** | −7.9 pt |

The ARC-Easy gain is what the FineWeb-Edu authors [[16]](#ref-16) report: filtering the web for educational content gives
["dramatically better performance on knowledge- and reasoning-intensive benchmarks like MMLU and
ARC"](https://arxiv.org/abs/2406.17557).

### Where the LAMBADA points went

The LAMBADA drop is a loss of a *capability* rather than of knowledge. LAMBADA passages come
from novels, and the target is the last word of the passage. The passages where GPT-2 beats Model A by the
widest margin all follow the same pattern:

> "…He passed the glasses to Bobby who had a look. "That looks like your boy the tow truck driver, doesn't it **Bobby**"
>
> "…she shifted her gaze to Gabe. "You know this through fabled stories, don't you **Gabe**"

The answer is a character's name, mentioned earlier in the passage, so the model only has to copy it. GPT-2 usually does; our models mostly don't.
Splitting all 5,153 passages by whether the target word is capitalized (almost always a
name) makes the pattern measurable:

![LAMBADA perplexity by target type](/assets/s1/headline_comparison/figures/lambada_targets.png)

| Perplexity on the target | Capitalized (2,439) | Lowercase (2,714) |
|---|---|---|
| Model A | 134.9 | 93.3 |
| Model B | 133.7 | 98.5 |
| GPT-2 | 27.1 | 57.0 |
| GPT-2's advantage over A | **5.0×** | 1.6× |

For GPT-2, names are *easier* targets than ordinary words; for our models they're the hardest. The
architecture can't explain that, since Model A and GPT-2 are architecturally identical, so the likely explanation is
the data. Dialogue that addresses characters by name is common in fiction, and WebText contains a lot of
fiction, because Reddit links to stories.
FineWeb-Edu filters most of it out, since fiction scores low on educational value, so our models rarely saw a name copied out of dialogue during training.

The mechanism for copying from context, induction heads, appears even in very small transformers
(Olsson et al., 2022) [[13]](#ref-13). So a bigger model trained on the same filtered data would have more capacity but no more
practice, and there's no reason to expect scale alone to close this gap. Changing the data might.

## A 50:50 data mix

To test that, the next run changes only the data. Model B's architecture
and token budget stay fixed; the training data becomes half FineWeb-Edu, half
[Cosmopedia-v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), a corpus of synthetic textbooks and
stories generated by a larger model. Hugging Face's SmolLM2 models [[17]](#ref-17) mix it
into their own pretraining data.

Mixing two datasets needs a sampling rule. The common one is random
sampling, where each batch picks a source with probability 0.5. That's 50:50 on average, but any particular step can
be lopsided, and replaying the exact same order after a restart means saving the random generator's state. This run
uses a deterministic rule instead, the one Megatron-LM [[12]](#ref-12) uses to blend datasets: each micro-batch goes to whichever
dataset is furthest below its target share so far. With two datasets at 50:50 that simply alternates, but the same
rule handles any weights (90:10, three datasets, etc.) and never drifts from them by more than one micro-batch. A restart only needs to know how many micro-batches each dataset has contributed.

Each dataset is read in order, so the 4.9B FineWeb-Edu tokens in this run
are exactly the first 4.9B that Model B saw. The
run reports loss on held-out FineWeb-Edu and held-out Cosmopedia-v2 from the start, since loss on FineWeb-Edu alone
would make any change of data look bad.

![Validation loss on both datasets during training](/assets/s1/data_mix/figures/loss_curves.png)

Before the run, I expected ARC-Easy to move most, since Cosmopedia is textbook-style science and
explanation; FineWeb-Edu loss to get worse than Model B's, because the model sees half as much of it; and Cosmopedia
loss to improve a lot. LAMBADA was the open question. Cosmopedia includes synthetic stories, so it might recover some
of the narrative capability FineWeb-Edu filtered out, or it might not, if synthetic stories don't use names the way novels
do.

The p-values and confidence intervals compare the 50:50 model with Model B, using the same paired tests as in the
Results section.

| | Model B, 100% FineWeb-Edu | Model B, 50:50 | Change |
|---|---|---|---|
| Held-out perplexity, FineWeb-Edu | 21.80 | 24.56 | +2.76 (+12.7%), 95% CI [+2.64, +2.89] |
| Held-out perplexity, Cosmopedia-v2 | 20.49 | **6.76** | −13.73 (−67%), 95% CI [−13.97, −13.50] |
| HellaSwag (acc_norm) | 32.5 | 33.2 | +0.7 pt, p = 0.04 |
| ARC-Easy (acc) | 51.9 | 51.6 | −0.3 pt, p = 0.78 |
| LAMBADA accuracy | 24.5 | **26.9** | +2.3 pt, p < 0.001 |

Giving up half the FineWeb-Edu costs 12.7% in perplexity on FineWeb-Edu, while
perplexity on Cosmopedia drops by two-thirds, to 6.8. Synthetic textbook text is very predictable once a
model has seen some of it, and a model this size learns its style almost completely.

The ARC-Easy prediction was wrong: it didn't move. One plausible explanation is that FineWeb-Edu already
supplies the kind of grade-school science ARC asks about, so more textbook prose adds little. HellaSwag rose by 0.7
points with p = 0.04, just under the 0.05 bar. But this table runs five tests, and the chance that at least one
crosses 0.05 by luck alone is about 23% (1 − 0.95⁵), so I treat it as a hint, not a result.

LAMBADA is the benchmark that clearly moved: accuracy rose 2.3 points,
and splitting by target word shows where:

| LAMBADA perplexity on the target | Capitalized (names) | Lowercase |
|---|---|---|
| Model B, 100% FineWeb-Edu | 133.7 | 98.5 |
| Model B, 50:50 | **84.1** | 115.3 |
| GPT-2 | 27.1 | 57.0 |

Cosmopedia's synthetic stories address characters by
name, so the mixed model got some practice at copying a name from earlier in the passage, the skill FineWeb-Edu
had filtered out. It's still 3x worse than GPT-2 on names, so synthetic fiction gives back only part
of the skill. Meanwhile the lowercase targets, closer to FineWeb-Edu's kind of text, got a little worse with
half the FineWeb-Edu gone. With architecture and token budget fixed, swapping half the data moved LAMBADA more than any architectural change in
this article, and in the direction the data's content predicts.

Larger projects tune data mixtures the same way, with many more runs. SmolLM2 was built on small ablations of
data mixtures, then trained in stages, with each stage's mixing rates adjusted based on the one before (Allal et al. [[17]](#ref-17)). NVIDIA's two-phase pretraining moves to higher-quality data late in training (Feng et al., 2024) [[18]](#ref-18). A single
50:50 data mixture run is the smallest version of that loop.  Other possible mixture strategies are blends that
changes as training iterations proceed.


## What should carry over, and what might not

Some of these findings follow from arithmetic and hold at any size; others are specific to 124M models and 1024-token contexts:

| Finding at 124M | With bigger models | With longer contexts | Grounded in |
|---|---|---|---|
| The architectures tie on loss; RoPE helps early | Unknown from these runs; the papers report GQA and local windows as close to quality-neutral by design | Local windows risk long-range quality; global layers carry it | Papers; the rest is open |
| Output layer is ~30% of FLOPs, attention ~7% | Output layer falls to a few percent | Attention surpasses the MLP at $$T ≈ 8d$$ | Arithmetic |
| Model B trains 4% slower despite fewer FLOPs | The gap should shrink: its extra work grows with $$d$$, matmuls with $$d^2$$ | Local windows start saving real FLOPs, given a kernel that skips masked blocks | Arithmetic + kernel behaviour |
| Data can be more important than architecture for certain capabilities | No reason to expect scale alone to fix a data gap | — | Our runs + Olsson et al. 2022 [[13]](#ref-13) |
| Most benchmarks sit near chance | They become informative, and paired tests matter more, since differences between strong models are small | — | Published results |
| Plain data parallelism: every GPU holds the whole model and optimizer | Tops out at ~4.5B parameters per 80 GB GPU (about 17 bytes per parameter); larger needs state sharding | — | Arithmetic (ZeRO [[10]](#ref-10)) |
| Run resume, storage and data replay need care | More critical: ~84 GB checkpoints at 7B; spikes need precise batch skipping | — | Arithmetic + PaLM [[11]](#ref-11) |




## Acknowledgement
Thanks for reading through my work and findings! If you spot a mistake, or have comments or critiques, I'd be glad to
hear them: please reach out on [LinkedIn](https://www.linkedin.com/in/shinibali-bhattacharyya/) or [X](https://x.com/ShinibaliB).

This article leans on people who explained things well and acted as my knowledge base:

- **Andrej Karpathy:** [nanoGPT](https://github.com/karpathy/nanoGPT),
  [build-nanogpt and *Let's reproduce GPT-2 (124M)*](https://github.com/karpathy/build-nanogpt), and
  [llm.c](https://github.com/karpathy/llm.c).
  The llm.c discussions are where the GPT-2 HellaSwag reference numbers come from.
- **Aleksa Gordić:** [*Inside the Transformer: The Life of a Token*](https://www.aleksagordic.com/blog/transformer).
  It's the reason Model B is Gemma 3-style, and it's the article to go to for an illustrated understanding
  of RoPE, YaRN, GeGLU, RMSNorm and hybrid local/global attention.
- **Percy Liang and Tatsunori Hashimoto:** Stanford's [*CS336: Language Modeling from Scratch*](https://cs336.stanford.edu/).
  Its lectures are a big part of how I learned the material behind this series.
- **Horace He:** [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html). It's the
  lens for every speed number here: whether an operation is limited by compute, memory bandwidth or overhead.
- **Hugging Face:** [*The Smol Training Playbook*](https://huggingfacetb-smol-training-playbook.hf.space/), on running
  ablations and training at the scale of hundreds of GPUs, and the
  [FineWeb](https://arxiv.org/abs/2406.17557) and SmolLM teams for FineWeb-Edu and Cosmopedia, the data both models
  learned from.

Platforms and tools I used:

- [RunPod](https://www.runpod.io/) provided the H100 GPUs, rented by the hour, for every training run.
- [Weights & Biases](https://wandb.ai/) logged every run. Most of the figures here are rebuilt from those logs.
- [Hugging Face Hub](https://huggingface.co/) stored the model checkpoints and the tokenized shards.
- EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) ran every benchmark.
- [Claude Code](https://claude.com/claude-code), Anthropic's coding assistant, worked alongside me throughout: it guided the GPU pod setup and checked each run, helped me debug when I was stuck, wrote the analysis and plotting scripts, and helped draft and fact-check this article.

## Appendix

### Counting the FLOPs

The forward FLOPs per token, from applying the $$2mn$$ rule to every matrix multiplication in a layer: the linear projections and the attention computation itself.

| Symbol | Meaning | Model A | Model B |
|---|---|---|---|
| $$d$$ | model dimension ($$d_{model}$$): the size of each token's vector | 768 | 768 |
| $$L$$ | layers (global + local) | 12 (12 + 0) | 12 (3 + 9) |
| $$n_h, n_{kv}$$ | query heads, key/value heads | 12, 12 | 12, 3 |
| $$h = d / n_h$$ | head size | 64 | 64 |
| $$T$$ | context length | 1024 | 1024 |
| $$W$$ | local window | — | 512 |
| $$V$$ | vocabulary (padded) | 50,304 | 50,304 |

Per layer, per token:

| Piece | FLOPs | Model A | Model B |
|---|---|---|---|
| Q, K, V projections | $$2d^2 + 2d·(2n_{kv}·h)$$ | $$6d^2$$ | $$3d^2$$ (GQA, $$n_h : n_{kv} = 4:1$$) |
| Attention output projection | $$2d^2$$ | $$2d^2$$ | $$2d^2$$ |
| MLP | GELU: $$2(d·4d + 4d·d)$$; GeGLU: $$2(d·2f + f·d)$$ with $$f = 8d/3$$ | $$16d^2$$ | $$16d^2$$ |
| Attention scores $$q·k$$ + weighted sum of $$v$$ | $$4d·k̄$$ | $$4d·(T + 1)/2$$ | same in global layers; $$4d·k̄_W$$ in local ones |

Here $$k̄$$ is the number of keys a query sees, averaged over positions. With causal masking, position $$p$$ sees $$p + 1$$
keys, so a global layer averages $$(T + 1)/2$$. A local layer ramps up to $$W$$ keys and then stays there, so $$k̄_W = [W(W + 1)/2 + (T − W)·W] / T$$.

Once per token, the output layer adds $$2dV$$ FLOPs.

Adding it up:

$$F_A = L\,\big(24d^2 + 2d(T+1)\big) + 2dV$$

$$F_B = 21\,L\,d^2 + 2d(T+1)\,L_{\text{global}} + 4d\,\bar{k}_W\,L_{\text{local}} + 2dV$$

The $$21d^2$$ assumes Model B's four query heads per key/value head, so $$n_{kv}·h = d/4$$. With other ratios, the projection
term is $$2d(d + 2·n_{kv}·h) + 2d^2$$.

Plugging in the numbers:

- Model A: $$12 × (14.16M + 1.57M) + 77.27M = 266.0M$$ FLOPs per token.
- Model B: $$12 × 12.39M + 3 × 1.57M + 9 × 1.18M + 77.27M = 241.2M$$ FLOPs per token ($$k̄_W = 384.25$$).

### Reproducing this article

The trained models are public on Hugging Face, each with a model card:
[Model A](https://huggingface.co/CuriosLM/gpt2-124m-fineweb-edu), in the standard 🤗 transformers GPT-2 format;
[Model B](https://huggingface.co/CuriosLM/gemma-style-113m-fineweb-edu); and
[Model B on the 50:50 mix](https://huggingface.co/CuriosLM/gemma-style-113m-fineweb-edu-cosmopedia).

The code for training, evaluation and every figure is in the [CuriosLM repository](https://github.com/Shinibali/CuriosLM),
and its README explains how to run each part.

### Compute and cost

| Session | Runs | Hardware | Training time | Cost |
|---|---|---|---|---|
| First pods | Setup and smoke tests | — | — | $4.01 |
| Calibration and ladder | Micro-batch calibration; 5 rungs × 1.5B tokens | 2× H100 SXM | ~2.5 h (ladder) | $22.36 |
| Seed rerun and headline | Rung a4 with a second seed; Models A and B, 9.8B tokens each | 2× then 4× H100 SXM | 0.5 h; 1.6 h + 1.7 h | $51.94  |
| Data mix | Model B on the 50:50 mix, 9.8B tokens; smoke test; H100 inference benchmark (Part 2) | 4× H100 SXM | 1.7 h | $29.49 |
| YaRN | Smoke test; 200-step fine-tune (Part 2) | 1× H100 SXM | ~0.4 h | $2.07 |
| **Total** | | | | **$109.87** |


### What broke during training and how I fixed it

None of the results above came out of a clean first attempt. These are the failures that cost real time or money, in no particular order:

- **Resumed runs weren't exact:** A run that stops and restarts should be indistinguishable from one that never stopped. After the first resume, the
loss dropped from 10.2 to 9.7 in a few steps, suspiciously fast: the model was seeing data it had already trained
on. Three separate bugs were behind it:

  - The data loader's position wasn't saved in the checkpoint, so a resumed run started the data over.
  - The batch the loader had already fetched ahead of time was skipped on restart.
  - Evaluation was drawing its batches from the training stream, which shifted where training read next.
  
  The fixes: save the loader's state with the checkpoint, restore the prefetched batch, and give evaluation
its own fixed batches. A test now stops a run, resumes it, and checks it matches an uninterrupted run bit for bit.

- **Local attention made training slower:** The first version of rung a4 trained 9% *slower* than the rung before it, because the
window, written as a dense mask, knocked attention off its FlashAttention path. FlexAttention's block masks won back
about 5%, and brought problems of their own. FlexAttention has no backward pass on Apple GPUs or CPUs, so it only runs
on CUDA under `torch.compile`, with the dense mask as the fallback; both paths read the window rule from one shared
function, so they can't disagree. Along the way, the mask got built as a tensor instead of a function, FlexAttention
got applied to the *global* layers too, and a mask buffer was never registered. Unit tests caught all three.

- **The data mix alternated by optimizer step:** The 50:50 run blends two datasets micro-batch by micro-batch. The first version of the blending loader decided the
mix the same way on every GPU. On one GPU that's correct. On 8 GPUs, all eight would read FineWeb-Edu on the same
micro-step, then all eight would read Cosmopedia on the next: every *optimizer step* would have been 100% one dataset
or the other, alternating. The fix numbers the micro-batches globally across GPUs and runs Megatron-LM's blending rule
over that global sequence. Every step is now exactly 50:50 on 1, 2, 4 or 8 GPUs, and the tests check each
count.

- **MMLU took three hours on the Mac:** Before setting MMLU aside, I ran it once to check whether a model this size could beat chance. That
run crawled on the M1 Pro, while HellaSwag had been fast. The process had grown to 17 GB and was swapping to
disk. MMLU's prompts are long, and scoring needs the full-vocabulary logits: one batch of 16 prompts × 1024 positions ×
50,304 vocabulary entries × 4 bytes is 3.3 GB. The log-softmax made a second copy and the model's own loss
computation a third. HellaSwag's short prompts never got close. A batch size of 4 fixed it; the better fix is to keep only the rows that are actually scored. Smoke tests should use the *longest* inputs instead of some random or first few samples.

- **NCCL failure at pod startup:** The first 2-GPU launch on RunPod died before training even started, with
`Failed to bind NVLink SHARP (NVLS) Multicast memory … CUDA error 401`. NVLS is an NCCL feature that some hosts don't
support when you rent a slice of an 8-GPU machine. Some hosts were fine; the 50:50 run's 4-GPU pod hit it again. Setting
`NCCL_NVLS_ENABLE=0` turns the feature off, which costs nothing at 2–4 GPUs and doesn't change the math.

These problems grow with model size. Our checkpoints are 12 bytes per parameter (float32 weights plus Adam's two
moment estimates). For comparison, a 7B checkpoint would be about 84 GB, close to our CuriosLM Hugging Face org's entire 100 GB cap. The same habits apply at any size: smoke-test every machine, verify every checkpoint, make the data order replayable.



## References

1. <span id="ref-1"></span>Radford et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
2. <span id="ref-2"></span>Brown et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).
3. <span id="ref-3"></span>Su et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
4. <span id="ref-4"></span>Ainslie et al. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245).
5. <span id="ref-5"></span>Shazeer (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).
6. <span id="ref-6"></span>Zhang & Sennrich (2019). [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).
7. <span id="ref-7"></span>Gemma Team (2024). [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118).
8. <span id="ref-8"></span>Gemma Team (2025). [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786).
9. <span id="ref-9"></span>Hoffmann et al. (2022). [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
10. <span id="ref-10"></span>Rajbhandari et al. (2019). [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054).
11. <span id="ref-11"></span>Chowdhery et al. (2022). [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311).
12. <span id="ref-12"></span>Shoeybi et al. (2019). [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053).
13. <span id="ref-13"></span>Olsson et al. (2022). [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895).
14. <span id="ref-14"></span>Dao et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).
15. <span id="ref-15"></span>Dong et al. (2024). [Flex Attention: A Programming Model for Generating Optimized Attention Kernels](https://arxiv.org/abs/2412.05496).
16. <span id="ref-16"></span>Penedo et al. (2024). [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557).
17. <span id="ref-17"></span>Allal et al. (2025). [SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model](https://arxiv.org/abs/2502.02737).
18. <span id="ref-18"></span>Feng et al. (2024). [Maximize Your Data's Potential: Enhancing LLM Accuracy with Two-Phase Pretraining](https://arxiv.org/abs/2412.15285).
19. <span id="ref-19"></span>Zellers et al. (2019). [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830).
20. <span id="ref-20"></span>Clark et al. (2018). [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457).
21. <span id="ref-21"></span>Paperno et al. (2016). [The LAMBADA dataset: Word prediction requiring a broad discourse context](https://arxiv.org/abs/1606.06031).
22. <span id="ref-22"></span>Miller (2024). [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations](https://arxiv.org/abs/2411.00640).
23. <span id="ref-23"></span>McNemar (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika* 12(2), 153–157. [doi:10.1007/BF02295996](https://doi.org/10.1007/BF02295996).
24. <span id="ref-24"></span>Efron (1979). Bootstrap Methods: Another Look at the Jackknife. *Annals of Statistics* 7(1), 1–26. [doi:10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552).
25. <span id="ref-25"></span>Korthikanti et al. (2022). [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198).
26. <span id="ref-26"></span>NVIDIA. [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html), sections "Tensor Core Requirements" and "Tile Quantization".
27. <span id="ref-27"></span>Ding et al. (2021). [CogView: Mastering Text-to-Image Generation via Transformers](https://arxiv.org/abs/2105.13290).

## Citation

```bibtex
@misc{bhattacharyya2026gpt_to_gemma_1,
  author       = {Bhattacharyya, Shinibali},
  title        = {GPT-2 (124M) to Gemma 3-Style (113M) language models, Part 1: Pretraining Lessons from Swapping One Architecture Component at a Time},
  year         = {2026},
  url          = {https://shinibali.github.io/gpt2-to-gemma-part-1/},
}
```
