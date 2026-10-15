---
layout: post
title: "GPT-2 (124M) → Gemma 3-Style (113M), Part 1: Pretraining Lessons from Swapping One Architecture Component at a Time"
date: 2026-10-14
math: true
---

In this article, I will share my learnings from pretraining language models from scratch and how to build intuition about their inner workings as model architecture and parameters scales. Thanks to epic learning resources available on the internet, I started with Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) project
[*Let's reproduce GPT-2 (124M)*](https://github.com/karpathy/build-nanogpt). I also got inspired by Aleksa Gordić's awesome illustrative article
[*Inside the Transformer: The Life of a Token*](https://www.aleksagordic.com/blog/transformer) where he walks through
Essential AI's Rnj-1.5 7B model, which closely follow the Gemma 3 architecture and uses YaRN to stretch their context
window. It explains grouped-query attention (GQA), rotary position
embeddings (RoPE), YaRN, gated MLP (GeGLU), RMSNorm and hybrid local/global attention so clearly that I wanted to find out: what do those model architecture choices actually buy when you train models at the scale I had started at, and how does data mixture strategy affect their capability? 

That question is answered in this two-part series: Part 1 (this article) is where I share learnings from building GPT-2 small (124M) Model A and then ablated its architecture with one component change at a time, finally obtaining a Gemma 3-style (113M) Model B, trained on the same 9.8B tokens of
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) while the tokenizer, the optimizer and the training tokens are all held fixed. [Part 2](/gpt2-to-gemma-part-2/) discusses how each model
represents position, visualises attention and induction heads, and stretches Model B's context 4x with YaRN. [Runpod](https://www.runpod.io/) H100 GPU cloud + storage cost spanning over ablation experiments, training, and fine-tuning for context-extension is just under $110, while local debugging, evals and analysis were conducted on M1 Mac Pro. Every number in this article comes from code in the [CuriosLM repository](https://github.com/Shinibali/CuriosLM), and every
figure can be regenerated from it. The trained models, with model cards, are on the Hugging Face org [CuriosLM](https://huggingface.co/CuriosLM).

The high-level findings:

- Both models tie on quality: Validation loss ends at 3.055 vs 3.051, and no benchmark difference is statistically
  significant. The Gemma recipe learns faster early on achieving lower validation loss, but by 10B tokens GPT-2 has caught up.
- While ablating from Model A to B, one component at a time, only RoPE improves quality at this size. Grouped-query attention, GeGLU with sandwich norm
  and local attention don't move the loss beyond seed noise.
- In theory Model B is cheaper to train, in practice it's slower. It needs 9% fewer FLOPs per token but trains 4%
  slower, because point-wise operations like the extra norms, gating and rotation are memory-bound work that FLOP counts don't see. Model FLOPs utilization (MFU) drops from 37% (Model A) to 32% (Model B) during training.
- At inference, Model B's grouped-query attention and local windows make its KV cache 6.4x smaller, so it can serve
  longer contexts and more sequences within the same memory footprint as Model A. At this model scale, whether that makes it generate faster
  turns out to depend on the hardware.
- Data matters more than architecture: Against the original GPT-2 [[1]](#ref-1), trained on different data, both models gain
  8 points on science questions (ARC-Easy) and lose 8 on story endings (LAMBADA). Most of that loss I could attribute to one capability: copying a character's
  name from earlier in the story.
- Changing the data gives back part of that capability: Swapping half the training data for synthetic
  textbooks and stories (Cosmopedia-v2) raised LAMBADA accuracy 2.3 points, almost entirely on character names, at
  the cost of 12.7% higher perplexity on FineWeb-Edu.

The specific results may not hold at larger scale but the measurement methodology should. 

## Setup: two different model architecture but same training tokens, tokenizer and optimizer

### Model architecture

Model A is GPT-2 small [[1]](#ref-1), exactly as [nanoGPT](https://github.com/karpathy/nanoGPT) built it. It has 12 layers ($$L$$), 12 attention heads, a model dimension
($$d_{model}$$ or simply $$d$$) of 768, and a 1,024-token context, with
learned absolute position embeddings, full multi-head attention in every layer, a GELU MLP, and LayerNorm before each
sub-layer.

Model B keeps the same skeleton (12 layers, 12 query heads, $$d$$  768, 1,024 tokens) and swaps in the Gemma 3-style
components. It is **not an exact copy of Gemma 3**, but only inspired the following architectural changes while keeping parameter count upper bounded to GPT-2 small.

A summary table of what each change is meant to do, what it did in these runs, and where to read more (the inference column is measured
in Part 2):

| Change | What it does | In training, here | At inference | Deep-dive references |
|---|---|---|---|---|
| RoPE | instead of learned positions, queries and keys are rotated by an angle so attention sees *relative* distance | The only change that improved loss (−0.085 at 1.5B tokens); 0.8M fewer parameters | No effect on cache size; makes context extension (YaRN) possible | Su et al. 2021 [[3]](#ref-3); Gordić's [article](https://www.aleksagordic.com/blog/transformer) |
| Grouped-query attention | 12 query heads share 3 key/value heads (4:1 Q-KV head ratio) | No measurable quality change; 8% fewer FLOPs per token; the fastest rung | 4× smaller KV cache | Ainslie et al. 2023 [[4]](#ref-4) |
| GeGLU + sandwich RMSNorm | a gated variant of GELU. A gated MLP has three weight matrices instead of two, so its inner width is cut to 2/3 of GPT-2's (2048 instead of 3072, which is $$4d$$) to keep the parameter count level. And RMSNorm both before and after each attention and MLP block, for smoother optimization | Within noise; same FLOPs (narrower MLP); slower, from extra elementwise operations | No effect | Shazeer 2020 [[5]](#ref-5); Zhang & Sennrich 2019 [[6]](#ref-6) |
| Local/global attention | 3 of every 4 layers only attend to the most recent 512 tokens. Every fourth layer sees the full context. So attention per layer looks like *GLLLGLLLGLLL*. | Within noise at a 1,024-token context; slower unless the kernel skips masked blocks | Cache stops growing at 511 positions in every 3 of 4 layers | Gemma 2 and 3 reports [[7]](#ref-7), [[8]](#ref-8) |

Original Gemma 3's local window is 1,024 tokens, which would cover
this model's entire 1,024-token context and make its local layers global, so Model B's window is half the
context, 512 tokens. It also keeps one global layer in every four rather than Gemma 3's one in six, and leaves
out Gemma 3's QK-norm, which I didn't test.

| | Model A (GPT-2 recipe) | Model B (Gemma 3 recipe) |
|---|---|---|
| Parameters | 124.5M | 113.1M |
| Position information | learned table, 1,024 × 768 | RoPE (no parameters) |
| Attention | 12 heads, all global | 12 query / 3 KV heads; 3 of 4 layers local (512) |
| MLP | GELU, 3,072 wide | GeGLU, 2,048 wide |
| Norm | LayerNorm, pre | RMSNorm, pre and post |

The two architectures side by side: GPT-2 and the Gemma 3-style recipe, as trained:
![The two architectures side by side: GPT-2 and the Gemma 3-style recipe, as trained](/assets/s1/figures/architecture.png)

Model B ends up with 9% fewer parameters, and almost all of the difference is GQA's smaller key/value projections. I
didn't enlarge $$d$$ to win them back, because the comparison holds the shape of the transformer blocks fixed and does 
not fix on the parameter total. At this size, majority of the model's parameters sits in the token
embedding table. Google's smallest Gemma [Gemma 3 270M](https://developers.googleblog.com/en/introducing-gemma-3-270m/) has about 170M parameters in its
embedding table, because its token vocabulary has 256K entries, and only about 100M in its transformer blocks. Our models
use GPT-2 tokenizer's 50K vocabulary, so their embedding table is far smaller, and Model B's transformer blocks (about 74M
parameters) put it in the same order of magnitude as Gemma 3 270M.

The embedding share is itself a small-model effect. With a vocabulary of $$V$$ entries, the embedding table holds
$$V·d$$ parameters, while the $$L$$ transformer blocks hold about $$12·L·d^2$$ ($$4d^2$$ in each
layer's attention projections from Q,K,V, and attention output and $$8d^2$$ from MLP for combined up and down projections). So the table's share falls quickly with size: 31% of GPT-2 small, 15% of GPT-2 medium, 5%
of GPT-2 XL. At the model scale of this article, which parameters you count really matters but at 7B or larger, it barely does.


### Data and tokenizer

Following nanoGPT's recipe, both models train on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (the `sample-10BT`
subset). Huggingface filtered web text by a ML classifier for educational value for this dataset and it is the strongest openly documented corpus for
small models. I only used GPT-2's tokenizer (50,257 entries, so $$V = 50{,}257$$) which is byte-level, so any text (emoji included) can be encoded without
"unknown" tokens. The data is tokenized once, into 100M-token shards (99 training and 1 validation). Each document is
prefixed with GPT-2's end-of-text token, `<|endoftext|>` (EOT), and the documents are concatenated back to back:

```
[EOT, doc 1 tokens…, EOT, doc 2 tokens…, EOT, doc 3 tokens…, …]
```

A shard is cut every 100M tokens, so a document that runs past the cut continues at the start of the next shard. The
EOT marks where one web page ends and the next begins, so the model learns not to read one into the other. Training
sequences are cut from this stream every 1,024 tokens, regardless of document boundaries.


### Training

Both models use nanoGPT's GPT-2 recipe unchanged: AdamW optimizer with a peak learning rate of 6e-4 decaying to 6e-5 on a
cosine schedule, $$\beta_2= 0.95$$, weight decay $$0.1$$ and gradient clipping at $$1.0$$. Training runs in bf16 with
`torch.compile` on rented H100s, with PyTorch's Distributed Data Parallel (DDP): every GPU holds a full copy of the
model, processes its own slice of the data, and the GPUs average their gradients before each optimizer step.

Three numbers dictate how much data a step sees. Each GPU processes a micro-batch of $$B = 60$$ sequences of $$T = 1{,}024$$
tokens at a time. Gradients from $$G = 8$$ micro-batches are accumulated before the optimizer updates the weights, and
$$G$$ is split across the GPUs (4 micro-batches each on 2 GPUs, 2 each on 4). So every optimizer step sees
$$B·T·G = 491{,}520$$ tokens, however many GPUs share the work.

That step size isn't tuned here. GPT-3 [[2]](#ref-2) trained its 125M model with batches of about 0.5M tokens, and
nanoGPT's GPT-2 config reaches ~0.5M as 491,520 tokens (12 sequences × 1,024 tokens × 40 micro-batches). Keeping it
keeps the optimization identical to that recipe, so only $$B$$ and $$G$$ were chosen for the hardware, with $$B·G = 480$$
sequences fixed.

Both models read the shards in the same order. On each micro-batch, the $$N_{GPU}$$ GPUs together read one contiguous
block of $$N_{GPU}·B·T$$ tokens from the current shard, GPU $$r$$ reading the $$r$$-th slice of $$B·T$$ tokens, and then all
of them stride forward by $$N_{GPU}·B·T$$ tokens. Over one optimizer step they cover 491,520 consecutive tokens.



## The ladder: one change at a time

### The tenet: ablate small, confirm big

Labs test core changes at small scale before spending real compute, and so does this project: a ladder of five 1.5B-token (~15% of the total training tokens) rungs
(about 0.5 X 2 GPU hours each on H100s), then the two ends of the ladder (GPT-2 vs Gemma 3 style) at the full 9.8B tokens (about 1.6 X 4 GPU hours each on 
H100s) with the same
data and the same hyperparameters. Every run's settings came from one model config change and couple of environment variables, and every run logged
to Weights & Biases. 

| Rung | Adds | Parameters | Val loss | Step vs previous |
|---|---|---|---|---|
| a0 | nothing: GPT-2 as nanoGPT ships it | 124.5M | 3.495 | |
| a1 | RoPE instead of learned positions | 123.7M | 3.410 | **−0.085** |
| a2 | grouped-query attention, 4 query heads per K/V head | 113.1M | 3.431 | +0.021 |
| a3 | GeGLU MLP, RMSNorm before *and* after each sub-layer | 113.1M | 3.454 | +0.023 |
| a4 | local attention (512-token window) in 3 of every 4 layers | 113.1M | 3.455 | +0.001 |

Each step is the difference between two single training runs, so it carries the run-to-run noise of both. To size
that noise, one rung (a4) was trained twice with different random seeds. The two copies ended 0.020 apart.

![Each rung's step against the seed noise band](/assets/s1/ladder/figures/ladder_loss.png)

Against that band, RoPE's step stands out: it is four times the seed-to-seed gap. It's the one change that makes the model
better at this scale, and it does so with *fewer* parameters, since the learned position table (0.8M parameters)
disappears. GQA and GeGLU each cost about 0.02 in val loss, right at the edge of the band. Either could be real or a coin flip,
and it would take two or three more seeds per rung to know. But I discontinued the extra runs to avoid diminishing returns on GPU hours. Local attention changes nothing measurable, which is what one would expect at a 1,024-token context, where a
512-token window still sees half of everything.

### What the other changes produce:

![Training throughput and KV-cache size per rung](/assets/s1/ladder/figures/ladder_efficiency.png)

The right panel is the point of GQA and local attention: GQA cuts the memory a model needs per sequence to generate text by 4x, from 75 to 19 MB per 1,024-token sequence
(with keys and values stored in float32, 4 bytes each, half that in bf16), because 12 query heads now share 3 key/value
heads. Local windows take it down to 12 MB (6.4x smaller than Model A's), because three layers in four keep only the last 511 positions. 

The left panel shows that the Gemma 3 recipe trains *slower*, at 820k tokens per second against GPT-2's 940k,
Two rungs each cost about 8.5% extra time: RoPE, whose rotations are extra elementwise FLOP on every query and key, and local
attention. Restricting attention to a window meant handing PyTorch's scaled dot-product attention function an
explicit mask, and PyTorch can't run its FlashAttention kernel [[14]](#ref-14) with an arbitrary mask. It falls back to a
general-purpose kernel that scores every query-key pair, including the ones the window then throws away.
FlexAttention [[15]](#ref-15) builds the window into a block mask instead, and skips the tiles of the attention matrix that
are fully masked. On the same two H100s, it cut rung a4's time per optimizer step from about 590 ms to about 560 ms
(about 5% faster, MFU 33.4% → 35.2%), but didn't close the whole gap. The ladder figure above shows the dense-mask
version but the final 9.8B-token Model B run used FlexAttention.

GQA, the one change that cuts training FLOPs by a meaningful amount (8%; the local windows cut another 1%), is the
fastest rung of all. So at 124M: one change for quality (RoPE), two for serving cost (GQA, local windows), and one
(GeGLU with sandwich norm) that this scale can't distinguish from noise. The next question is whether RoPE's effect survives a much longer run.

At larger scale of billions of model parameters, the papers/observations below report the same split between quality and cost:

- The GQA paper (Ainslie et al. 2023) [[4]](#ref-4) finds GQA close to full MHA quality at large scale,
  at close to multi-query speed.
- Gemma 3's report [[8]](#ref-8) describes its move to more local layers and shorter windows as cutting KV-cache memory with little
  effect on perplexity.
- RoPE is now the default in almost every open model.
- GLU variants are where the ladder and the literature differ: Shazeer (2020) [[5]](#ref-5) reports lower perplexity from gated MLPs
  in T5 pretraining, while here the change sits at the edge of seed noise.

## Confirming at 9.8B tokens

A ladder at 1.5B tokens can be misleading as a change that helps early can flatten out once the model has seen enough
data. So I trained Model A vanilla GPT-2 and Model B, the full Gemma 3-style recipe. Each run is 20,000 steps of 491,520 tokens, which comes to 9.83 billion FineWeb-Edu tokens, a round number of steps that stays just inside the 99 training shards (9.9B tokens): a single pass with no
repeated data. That's roughly 4x the "compute-optimal" budget the Chinchilla scaling law [[9]](#ref-9) suggests for our model size.

Validation loss of both models over 9.8B tokens, and the gap between them:
![Validation loss of both models over 9.8B tokens, and the gap between them](/assets/s1/headline_comparison/figures/loss_overlay.png)

The loss used for pretraining runs is the cross-entropy: the average negative natural log of the probability the model gave each correct next
token, over $$N$$ tokens. Because it uses the natural log, its unit is the *nat*. Perplexity is the same quantity exponentiated, and can be interpreted as the number of tokens the model is effectively choosing between:

$$\text{loss} = -\frac{1}{N}\sum_{i=1}^{N} y_i \ln p(x_i \mid x_{<i}) \qquad \text{perplexity} = e^{\text{loss}}$$

Both curves start near $$ln (V) = ln (50257) ≈ 10.8$$ for the untrained model, the cross-entropy loss of guessing uniformly over GPT-2 tokenizer's
50,257 tokens. Train and validation loss track each other the whole way, as one'd expect when no example is seen twice. The first 0.5B tokens cuts the loss by 2.75x, beyond which the loss slowly saturates. The bottom panel zooms in on the gap between the two model's val loss.

Model B's RoPE advantage shows up immediately. After 0.5B tokens, B is ahead by 0.196, a huge
margin at this point in training. Then the gap closes. By about 4.4B tokens it's inside the seed-noise band, and at the
end it's 0.005, with B at 3.0506 and A at 3.0553. That's a tie in quality as the gap is a quarter of the difference between two
seeds of the same model. This data helps conclude that at this size, the Model B's architecture brings early progress, but does not provide a better endpoint. Whether the two endpoints separate again at
larger scale is a question these runs can't answer.


## Training on a budget: FLOPs, MFU and what a run costs

### Counting FLOPs

The counts below are for the forward pass: what it costs to run one token through the model, which is also what
inference pays per token. Training adds a backward pass, which costs about twice the forward, so about 3x in total.

Almost all of a transformer's arithmetic is matrix multiplication. Multiplying a token's $$m$$-dimensional vector by an
$$m × n$$ weight matrix costs ~$$2mn$$ FLOPs: one multiply and one add per weight. Applied to each piece of a
layer gives the forward cost per token. It gives **266M** FLOPs per token for Model A and **241.2M** for Model B. The full
derivation is in the appendix ([Counting the FLOPs](#counting-the-flops)). Here, $$d$$ is the
model dimension, $$T$$ the context length and $$k̄$$ the average number of keys a query sees.

The count leaves out everything that isn't a matrix multiplication: biases, normalization, activations, softmax and
masking, RoPE's rotations, residual additions, the embedding lookup and the loss. By a rough count they add up to under 1% of
the FLOPs, but they take far more than 1% of the time: each one reads and writes whole tensors from memory while
doing very little arithmetic, as [How well the GPUs were used](#how-well-the-gpus-were-used) shows below.

The algebra explains the results below:

- GeGLU costs exactly what GELU costs: With the hidden width trimmed to $$8d/3$$, both MLPs land at $$16d^2$$ FLOPs. Any
  difference between rungs a2 and a3 can't be a compute difference.
- GQA's saving is the $$6d² → 3d²$$ term, $$3d²$$ per layer. That's 21.2M FLOPs per token, 8 out of 9 points of
  FLOPs reduction in Model B. The local windows supply the last point (3.5M), by shortening $$k̄$$ in nine layers.
- Attention is the only term that grows with context length. Per layer, the attention scores cost $$2d(T + 1)$$
  against $$24d^2$$ for the weight matrices, a ratio of $$(T + 1)/(12d)$$. At $$T = 1{,}024$$ and $$d = 768$$ that's 11%, so
  attention is about $$10\%$$ of each layer's FLOPs (about $$7\%$$ of the whole forward pass, once the output layer is
  counted). So at this size and context, local windows don't save many FLOPs; their payoff is in the KV cache.
- Without the attention term, the forward pass is 2 FLOPs per weight, and
  training (the forward pass, plus a backward pass costing about 2x the forward) is $$2 + 4 = 6$$ FLOPs per weight,
  the well-known $$6N$$. The attention term adds
  $$6·L·d·(T + 1)$$ on top.

### Where the FLOPs go

Forward FLOPs (millions) per token, by component:
![Forward FLOPs per token, by component](/assets/s1/efficiency/figures/flops.png)

One thing worth noticing at this model scale: the output layer is almost 1/3 of the total compute. Projecting onto a
50,304-word vocabulary costs 2/3 as much as the whole MLP stack. 
Training costs about 3x the forward pass: 798 MFLOP per token for Model A and 724 for Model B. Over 9.83 B tokens, that's $$7.8 × 10¹⁸$$ and $$7.1 × 10¹⁸$$ FLOPs. The plain $$6N$$ rule comes within 7% of these numbers and the
difference is due to the attention term.

### How well the GPUs were used

Model FLOPs utilization (MFU) is the FLOPs a run actually achieves per second, divided by what the hardware promises to do
at peak. For an H100 SXM in bf16, the promised peak is 989 TFLOP/s: [NVIDIA's spec sheet](https://www.nvidia.com/en-us/data-center/h100/)
lists 1,979 TFLOP/s "with sparsity", and dense matrix multiplication gets half that.

| Headline runs, 4× H100 | Model A | Model B |
|---|---|---|
| Throughput | 1.82M tokens/s | 1.76M tokens/s |
| MFU, as the training script logs it (PaLM formula) | 39.4% | 35.1% |
| MFU, counting only useful FLOPs | 36.8% | 32.1% |
| GPU-hours for 9.83B tokens | 6.0 | 6.2 |

Following the initial nanoGPT implementation, the logged MFU uses the PaLM paper's [[11]](#ref-11) formula, $$12·L·d·T$$ for attention, which charges every token for attending to
the whole context. Corrected attention FLOPs for causal and windowed model gives a few points
less MFU. B loses more because the PaLM formula overcounts its local
layers for attention they skip. The gap is small at 1,024 tokens and grows with context. [Part 2](/gpt2-to-gemma-part-2/) works out by how much.

MFU across the two 9.8B-token runs, as logged and counting useful FLOPs only:
![MFU across the two 9.8B-token runs, as logged and counting useful FLOPs only](/assets/s1/efficiency/figures/mfu.png)

The training script logs a running average of MFU, in which each new step's reading counts for 10%, and it leaves
out the steps that also ran an evaluation or saved a checkpoint. So the remaining dips are real stretches where steps
ran slower, for reasons the logs don't record. It is worth noticing that Model B does 9% fewer FLOPs per token, yet it runs 4% slower. So it spends more time on work that isn't matrix multiplication:
- two extra normalizations per block;
- the gating multiply in GeGLU;
- the rotations in RoPE;
- the attention kernel's handling of the local windows.

Each of these is cheap in FLOPs but has to read and write whole activation tensors from memory. In [Horace He's words](https://horace.io/brrr_intro.html),
they're memory-bandwidth-bound not compute-bound. FLOP counts alone don't predict training speed, and a model that
looks cheaper on paper can be slower in practice.

### How the balance shifts as models and contexts grow

The equations in [Counting the FLOPs](#counting-the-flops) says how each components FLOPs shares move with scale. In the plot below, the bars show the output layer shrinking from 29% of the forward pass at
124M to 5% at 1.6B and about 3% at 6.7B, while the MLP becomes the majority at fixed context length $$T$$. The MLP and attention
projections grow with $$d²·L$$, the output layer with $$d·V$$, and attention scores
with $$d·T·L$$, the only term that depends on context. Attention overtakes the MLP at $$T ≈ 8d$$, about 6000 tokens at our
$$d$$ of 768, and [Part 2](/gpt2-to-gemma-part-2/) follows that term to long contexts. The crossover that matters at this
size is the output layer: it outweighs all the MLPs together while $$d·L < V/8$$. With GPT-2 tokenizer's 50304-word
vocabulary, $$V/8 ≈ 6300$$. Our models sit at $$d·L = 9216$$, just past the crossover, which is why the output layer is
2/3 of the MLP cost here. A model with Gemma 3 tokenizer's 262,144-word vocabulary needs $$d·L$$ above about 33,000 before
its MLPs outweigh its output layer.

![Each component's share of a model's forward FLOPs, from 124M to 6.7B parameters](/assets/s1/efficiency/figures/flops_by_size.png)

### Getting the GPUs busy in the first place

The first runs on the H100s showed an MFU of about 25% with a micro-batch $$B$$ of 12 sequences per GPU. One easy fix was simply to give each step more compute work by increasing the micro-batch. A
124M model is small enough that, with 12 sequences per GPU at a time, a large share of every step goes to fixed
overheads: launching kernels, moving small tensors, synchronizing. Raising the micro-batch to 60 sequences, with
gradient accumulation reduced to keep total tokens per iteration constant at 491,520 tokens, took MFU up to 41%. The optimizer update is mathematically identical, since
every step still sees 491,520 tokens. Beyond 60 sequences the gain flattened out while
memory kept climbing. And because the $$G$$ accumulated micro-batches are split across the GPUs, moving from 2 to 4 GPUs sped training up 1.94x with identical optimizer steps.

The training data never sits in GPU memory. The 100 tokenized shards (100M tokens × 2 bytes each) take 20 GB of disk.
They live on our Hugging Face Hub, each pod downloads them in about a minute, and the data loader keeps only the current
200 MB shard in CPU memory. At scale, data becomes a storage problem of its own: a 15T+ token corpus is 30–60 TB
once tokenized (2 to 4 bytes per token), too much to copy onto every machine, so it is streamed from shared storage instead.

Right now is a good point to dive a bit deeper into GPU resource estimation math. Each GPU's memory (80 GB of HBM on an H100) holds two main things: the **model state**, which grows with the parameter count
$$N ≈ 12·L·d^2 + V·d$$, and the **activations** saved for the backward pass, which grow with the micro-batch of $$B$$
sequences of $$T$$ tokens. We note their estimates from literature survey along with Model A's measured peak memory usage:

| Per GPU | Bytes | Model A, $$B = 60$$, $$T = 1024$$ |
|---|---|---|
| Model state: bf16 weights and gradients, float32 master weights, AdamW's two moments [[10]](#ref-10) | $$16N$$ | 2.0 GB |
| Activations in the transformer blocks [[25]](#ref-25) (omitting FlashAttention as it never stores the $$T × T$$ scores) | $$34·B·T·L·d$$ | 19.3 GB |
| Activations of the output layer: float32 logits for the loss (4 bytes each [[25]](#ref-25)) plus the bf16 logits the layer outputs (2 bytes) | $$6·B·T·V$$ | 18.5 GB |
| Estimated | | 39.8 GB |
| Measured peak | | 41.4 GB |

At this size, the output layer's activations are as large as all twelve transformer blocks', the same small-model effect as in
the FLOPs. As the blocks hold $$N ≈ 12·L·d^2$$
parameters, so their activations, $$34·B·T·L·d$$ are about $$17N · B·T/(6d)$$, and approximating that as $$16N · B·T/(6d)$$ while introducing a mere 6% error margin, we get

$$\text{training memory M} ≈ 16N\left(1 + \frac{B·T}{6d}\right) + 6·B·T·V $$

The activations equal the model state once a micro-batch holds about $$6d$$
tokens. Small models
are limited by activations, large ones by their state. But the activations are the flexible part: a smaller micro-batch $$B$$, made up with more gradient accumulation,
keeps the same 491,520-token optimizer step, and activation checkpointing recomputes activations instead of storing
them. The model state can't shrink, so it sets the ceiling. Turned around, the equation above gives the largest model a GPU with memory $$M$$ can train. With the smallest micro-batch,
$$B = 1$$ sequence of $$T = 1{,}024$$ tokens, and a width typical of models this size ($$d ≈ 3{,}000$$), $$B·T/(6d) ≈ 0.06$$ and
the output layer's $$6·B·T·V$$ is only 0.3 GB, so for an 80 GB H100:

$$N_{max} ≈ \frac{M - 6·B·T·V}{16\left(1 + \frac{B·T}{6d}\right)} ≈ \frac{80\text{ GB} - 0.3\text{ GB}}{16 × 1.06} ≈ 4.7\text{B}$$

Leaving a few GB for the CUDA context and workspace, which the equation doesn't
count, brings it to about 4.5B. Larger models have to be split across GPUs: ZeRO [[10]](#ref-10) (or PyTorch's FSDP) shards the optimizer state, gradients and weights across the data-parallel GPUs; tensor parallelism splits each layer's weight matrices across GPUs (Megatron-LM [[12]](#ref-12)); and pipeline parallelism places different layers on different GPUs.



## Interlude: the step-320 spike

While monitoring the first ladder runs, a sharp spike in training loss showed up at step 320. Training loss on a single batch is noisy, and spikes happen but this spike appeared in all the other runs at the same step:  both
architectures, short runs and long ones, on 2 GPUs and on 4. The only thing these runs share is the order of the
training data, so that gave way to the hypothesis that the problem is in the data.

![The training loss of three different runs around step 320](/assets/s1/step320_spike/figures/loss_spike.png)

Because the data loader is deterministic, the batch can be replayed exactly. Rebuild the data loader with the run's
settings, step it forward to the micro-batch that was logged at step 320, and score each of its 60 sequences. One
sequence stands far above the rest, with a mean loss of 5.06 against a batch median of 3.83.

It's an encyclopedia-style page of educational content, but it had
some words swapped by a ["baby talk" text converter](https://stackoverflow.com/beta/challenges/79640866/challenge-1-implement-a-text-to-baby-talk-translator-completed): *"a totaw of", "bewow de age of 18 years", "broders and cwose rewatives",
"de medod used".*

The converter swaps *l* for *w* and *th* for *d* (and, more rarely, a word-final *th* for *f* and *qu* for *qw*). For the model it's close to a
different language, and the tokenizer is where it breaks. `close` is one token, but `cwose` is
three (`c·w·ose`), each of them rare in that position. Other words turn into different words: `the` becomes ` de`, a
perfectly common token that means something else entirely, so the model has to predict a real word in the wrong place.

![Loss on each mangled word, real spelling vs baby talk](/assets/s1/step320_spike/figures/word_losses.png)

Comparing the page with a cleaned-up copy shows where the extra loss lives. 11 mangled words in the excerpt
account for 135 of its 165 extra nats. The other 30 spill onto the words after them: because as the model sees
"bewow de", even an ordinary "age" becomes harder to predict.

The cleaned copy came from a model trained in this study: the 1.5B-token checkpoint of ladder rung a4. For every word
containing a *w*, *d*, *qw* or word-final *f*, I generated every spelling the converter's rules could have mangled
into it (each *w* may be a real *w* or an *l*, each *d* a real *d* or a *th*, and so on), and let the checkpoint pick
the one it finds most likely after the already-corrected text before it. That fixed 196 words
across the page, with one miss: a foreign word with a genuine *d* came out with a *th*, because the model had no way
to know the *d* was real.

Pages like this are in every web corpus. The model learns to ignore it though. The loss is back to normal on the next step and no run shows lasting
damage, because the page is one of 60 sequences in the logged micro-batch (which is why the spike shows up) but
only one of 480 in the optimizer step. At GPT-3's [[2]](#ref-2) largest batch of about 3.2M tokens, it would be one in about 3,000.
So at larger scale, spikes tend to come from the optimization itself rather than from a single bad document, and
they're harder to ignore. The PaLM paper [[11]](#ref-11) reports about 20 spikes while training its 540B model. Its fix was to
restart from a checkpoint roughly 100 steps before each spike and skip the 200–500 data batches around it. That only
works if one can pin-point exactly which batches those were, which is what a deterministic and reproducible data loader gives
you. That is also what made this diagnosis quick.

## Evaluation: a tie on every benchmark

Validation loss says the two models are the same. But loss is an average log probability over every token, and two models can reach the same
average by being good at different things. So I evaluated both models through a set of standard benchmarks, with OpenAI's original GPT-2 (124M) checkpoint [[1]](#ref-1)
([`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) on Hugging Face), called "stock GPT-2" from
here on: the released weights, used as they are, alongside as a reference. All three go through the same harness, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
connected to the models through a small custom adapter.

### Choosing benchmarks for a 124M model

A benchmark is only useful if the model scores clearly above chance and below the ceiling. Otherwise every
model lands on the same number and the comparison tells us nothing. Inspired by some of the typical benchmarks used for measuring small model performance, four measures provided statistical signal, each for a reason:

- HellaSwag [[19]](#ref-19): Each sample has 4 multiple choice options that asks to pick the sensible ending to an everyday scenario. It's the standard yardstick for small models. GPT-2-sized models score around 30%, against 25% for random guess. Each ending is scored by its average log-probability  per character, so a long ending isn't penalized just for being long (lm-eval's `acc_norm`).
- ARC-Easy [[20]](#ref-20): grade-school science questions. These are knowledge-heavy, so this is where training data should show.
- LAMBADA [[21]](#ref-21): Each sample is to predict the last word of a passage, where the word can only be guessed from the wider passage. This set is the closest to a test of long-range context, which is where the two model's position schemes differ.
- Perplexity on held-out text: FineWeb-Edu (the training distribution) and Cosmopedia-v2 (synthetic textbooks). There are no question-answer format. This tests how well each model predicts real text and is the least noisy number available.

MMLU is left out because models this size sit at chance (25%), so it can't tell two of them apart.

### Results

Stock GPT-2 is trained on different data (WebText) so it is here just for reference.

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
Run-to-run (seed) noise is measured separately, from two seeds of one ladder rung)

**None of the benchmark gaps between A and B is significant.** On HellaSwag, for example, A and B gave different
answers on 841 of the 10,042 questions. A got 417 of those right and B got 424: a coin-flip split.

With perplexity score, B is 0.09 better on FineWeb-Edu and 0.18 worse on Cosmopedia-v2, and both
confidence intervals exclude zero. That makes both differences real *for these two checkpoints*, but only in the
sense that a different sample of test documents wouldn't change them. Retraining either model with a different
random seed moves its loss by about ±0.02, roughly ±0.4 perplexity, which is two to four times larger than either
gap. Each checkpoint wins one set and loses the other.

The GPT-2 row is the check for trusting the eval harness. An independent lm-eval run of the same GPT-2 checkpoint,
published on [AMD's model card for AMD-Llama-135m](https://huggingface.co/amd/AMD-Llama-135m), matches this result to within
0.001 on ARC-Easy, HellaSwag and LAMBADA. Both models beat GPT-2 by 8 points on ARC-Easy and lose to it by 8 on LAMBADA. Same size,
different data. That's the topic of next section.

At larger scale, benchmarks that sit at chance here, like MMLU, become informative, and paired tests matter more,
since differences between strong models are small.


## Data shapes capability more than architecture

Changing the architecture did not affect the evaluation outcomes significantly but changing the data moves outcome a lot. Stock GPT-2 is the evidence here: it's the
same size and architecture as Model A but was trained on different training data. Stock GPT-2 was trained on WebText, pages linked from Reddit (2019); Models A and B
on FineWeb-Edu, web pages scored for educational value (2024).

Against GPT-2, both models gain on some tasks and lose on others, by large margins and in the same direction:

| vs stock GPT-2 | Model A | Model B |
|---|---|---|
| ARC-Easy (grade-school science) | **+8.6 pt** | +8.1 pt |
| HellaSwag (everyday scenarios) | +1.3 pt | +1.3 pt |
| LAMBADA accuracy (last word of a story) | **−8.2 pt** | −7.9 pt |

The ARC-Easy gain is the one the FineWeb-Edu authors [[16]](#ref-16) advertise: filtering the web for educational content gives
["dramatically better performance on knowledge- and reasoning-intensive benchmarks like MMLU and
ARC"](https://arxiv.org/abs/2406.17557).

### Where the LAMBADA points went

The LAMBADA loss is the more interesting one, because it's a loss of a *capability*, not of knowledge. LAMBADA passages come
from novels, and the target is the last word of the passage. Looking at the passages where GPT-2 beats Model A by the
widest margin shows the same pattern every time:

> "…He passed the glasses to Bobby who had a look. "That looks like your boy the tow truck driver, doesn't it **Bobby**"
>
> "…she shifted her gaze to Gabe. "You know this through fabled stories, don't you **Gabe**"

The answer is a character's name, mentioned earlier in the passage, and the model only has to copy it. GPT-2 does.
Our models mostly don't. Splitting all 5,153 passages by whether the target word is capitalized (almost always a
name) makes the pattern measurable:

![LAMBADA perplexity by target type](/assets/s1/headline_comparison/figures/lambada_targets.png)

| Perplexity on the target | Capitalized (2,439) | Lowercase (2,714) |
|---|---|---|
| Model A | 134.9 | 93.3 |
| Model B | 133.7 | 98.5 |
| GPT-2 | 27.1 | 57.0 |
| GPT-2's advantage over A | **5.0×** | 1.6× |

For GPT-2, names are the *easy* targets, easier than ordinary words. For our models they're the hardest. Nothing in
the architecture explains that, since Model A and GPT-2 are architecturally identical. The likely explanation is in
the data. Dialogue that addresses characters by name is commonplace in fiction, and WebText is full of
fiction, because Reddit links to stories.
FineWeb-Edu has filtered most of it away since educational value scores fiction low. Our models never practised copying a name out of dialogue, so they don't.

The mechanism for copying from context, induction heads, appears even in very small transformers
(Olsson et al., 2022) [[13]](#ref-13). So a bigger model trained on the same filtered data would have more capacity but no more
practice, and there's no reason to expect scale alone to close this gap. Next section tests the other fix: changing the
data.

At this scale, *what* the model reads decides *what* it can do far more than how its attention is wired. So what
happens when half of it is swapped for something else?

## A 50:50 data mix

If data matters more than architecture, the natural experiment is to change only the data. For this experiment, we only picked Model B's architecture
and token budget stay fixed. Only what it reads changes: half FineWeb-Edu, half
[Cosmopedia-v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), a corpus of synthetic textbooks and
stories generated by a larger model. Hugging Face's SmolLM2 models [[17]](#ref-17) mix it
into their own pretraining data.

There's a small design choice hiding in the data mixture: how do you mix two datasets? The common answer is random
sampling, where each batch picks a source with probability 0.5. That's 50:50 on average, but any particular step can
be lopsided, and replaying the exact same order after a restart means saving the random generator's state. This run
uses a deterministic rule instead, the one Megatron-LM [[12]](#ref-12) uses to blend datasets: each micro-batch goes to whichever
dataset is furthest below its target share so far. With two datasets at 50:50 that simply alternates, but the same
rule handles any weights (90:10, three datasets etc) and never drifts from them by more than one micro-batch. A restart only needs to know how many micro-batches each dataset has contributed.

Each dataset is read in order, so the 4.9B FineWeb-Edu tokens in this run
are exactly the first 4.9B that Model B saw. And the
run reports loss on held-out FineWeb-Edu and held-out Cosmopedia-v2 from the start, since loss on FineWeb-Edu alone
would make any change of data look bad.

![Validation loss on both datasets during training](/assets/s1/data_mix/figures/loss_curves.png)

Some educated guesses can be made before seeing the results: ARC-Easy should move most, since Cosmopedia is textbook-style science and
explanation. FineWeb-Edu loss will be worse than Model B's, because the model sees half as much of it. Cosmopedia
loss will improve a lot. The open question is LAMBADA. Cosmopedia includes synthetic stories, so it might recover some
of the narrative capability FineWeb-Edu filtered out, or it might not, if synthetic stories don't use names the way novels
do.




| | Model B, 100% FineWeb-Edu | Model B, 50:50 | Change |
|---|---|---|---|
| Held-out perplexity, FineWeb-Edu | 21.80 | 24.56 | +2.76 (+12.7%), 95% CI [+2.64, +2.89] |
| Held-out perplexity, Cosmopedia-v2 | 20.49 | **6.76** | −13.73 (−67%), 95% CI [−13.97, −13.50] |
| HellaSwag (acc_norm) | 32.5 | 33.2 | +0.7 pt, p = 0.04 |
| ARC-Easy (acc) | 51.9 | 51.6 | −0.3 pt, p = 0.78 |
| LAMBADA accuracy | 24.5 | **26.9** | +2.3 pt, p < 0.001 |

The perplexity trade is lopsided. Giving up half the FineWeb-Edu costs 12.7% in perplexity on FineWeb-Edu. In return,
perplexity on Cosmopedia drops by two-thirds, to 6.8. Synthetic textbook text turns out to be very predictable once a
model has seen some, and a model this size learns its style almost completely.

The prediction about ARC-Easy was wrong: it didn't move at all. One plausible explanation is that FineWeb-Edu already
supplies the kind of grade-school science ARC asks about, so more textbook prose adds little. HellaSwag rose by 0.7
points, but with p = 0.04 and several tests run at once, that's suggestive.

The clear winner is LAMBADA, the benchmark this work traced to a missing capability. Accuracy rose 2.3 points,
and splitting by target word shows where:

| LAMBADA perplexity on the target | Capitalized (names) | Lowercase |
|---|---|---|
| Model B, 100% FineWeb-Edu | 133.7 | 98.5 |
| Model B, 50:50 | **84.1** | 115.3 |
| GPT-2 | 27.1 | 57.0 |

Cosmopedia's synthetic stories address characters by
name, so the mixed model got some practice at copying a name from earlier in the passage, the exact skill FineWeb-Edu
had filtered out. It's still 3x worse than GPT-2 on names, so a synthetic imitation of fiction gives back only part
of the skill. Meanwhile the lowercase targets, closer to FineWeb-Edu's kind of text, lost a little with
half the FineWeb-Edu gone.

With architecture and token budget fixed, swapping half the data moved LAMBADA more than any architectural change in
this article, and it moved one capability in the direction the data's content predicts.

At production scale, labs run this same kind of experiment, many times over. SmolLM2 was built on small ablations of
data mixtures, then trained in stages, with each stage's mixing rates adjusted based on the one before (Allal et al. [[17]](#ref-17)). NVIDIA's two-phase pretraining moves to higher-quality data late in training (Feng et al., 2024) [[18]](#ref-18). A single
50:50 point is the smallest version of that loop. Other possible mixture strategies are blends that
changes during training.


## What should carry over, and what might not

Some mathematical findings holds at all model sizes, others findings are scale specific. My understandings stands as follows:

| Finding at 124M | With bigger models | With longer contexts | Grounded in |
|---|---|---|---|
| The architectures tie on loss; RoPE helps early | Unknown from these runs; the papers report GQA and local windows as close to quality-neutral by design | Local windows risk long-range quality; global layers carry it | Papers; the rest is open |
| Output layer is ~30% of FLOPs, attention ~7% | Output layer falls to a few percent | Attention surpasses the MLP at $$T ≈ 8d$$ | Arithmetic |
| Model B trains 4% slower despite fewer FLOPs | The gap should shrink: its extra work grows with $$d$$, matmuls with $$d²$$ | Local windows start saving real FLOPs, given a kernel that skips masked blocks | Arithmetic + kernel behaviour |
| Data, not architecture, decides certain capabilities | No reason to expect scale alone to fix a data gap | — | Our runs + Olsson et al. 2022 [[13]](#ref-13) |
| Most benchmarks sit near chance | They become informative, and paired tests matter more | — | Published results |
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
  It's the reason Model B is Gemma 3-style and Part 2 ends with YaRN, and it's the article to-go for illustrative understanding
  of RoPE, YaRN, GeGLU, RMSNorm and hybrid local/global attention.
- **Horace He:** [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html). It's the
  lens for every speed number here: whether an operation is limited by compute, memory bandwidth or overhead.
- **Hugging Face:** [*The Smol Training Playbook*](https://huggingfacetb-smol-training-playbook.hf.space/), on running
  ablations and training at the scale of hundreds of GPUs, and the
  [FineWeb](https://arxiv.org/abs/2406.17557) and SmolLM teams for FineWeb-Edu and Cosmopedia, the data both models
  learned from.

Platforms and tools used:

- [RunPod](https://www.runpod.io/) provided the H100 GPUs, rented by the hour, for every training run.
- [Weights & Biases](https://wandb.ai/) logged every run. Most of the figures here are rebuilt from those logs.
- [Hugging Face Hub](https://huggingface.co/) stored the model checkpoints and the tokenized shards.
- EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) ran every benchmark.
- [Claude Code](https://claude.com/claude-code), Anthropic's coding assistant, worked alongside me throughout: guiding the GPU pod setup and checking each run; helped with debugging and troubleshooting when I was stuck; writing the analysis and plotting scripts; helping draft and fact-check this article.

## Appendix

### Counting the FLOPs

The derivation behind total FLOPs where the $$2mn$$ rule is applied to each piece of a layer involving matrix multiplications due to linear projections or simply attention computation.

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
| Q, K, V projections of input tokens | $$2d² + 2d·(2n_{kv}·h)$$ | $$6d²$$ | $$3d²$$ (GQA $$4:1=n_h: n_{kv}$$) |
| Attention output projection | $$2d²$$ | $$2d²$$ | $$2d²$$ |
| MLP | GELU: $$2(d·4d + 4d·d)$$; GeGLU: $$2(d·2f + f·d)$$ with $$f = 8d/3$$ | $$16d²$$ | $$16d²$$ |
| Attention Scores $$q·k$$ + weighted sum of $$v$$ | $$4d·k̄$$ | $$4d·(T + 1)/2$$ | same in global layers; $$4d·k̄_W$$ in local ones |

Here $$k̄$$ is the number of keys a query sees, averaged over the window. With causal masking, position $$p$$ sees $$p + 1$$
keys, so a global layer averages $$(T + 1)/2$$. A local layer ramps up to $$W$$ keys and then stays there, so
$$k̄_W = [W(W + 1)/2 + (T − W)·W] / T$$. 

Once per token, the output layer adds $$2dV$$ FLOPs.

Adding it up:

$$F_A = L\,\big(24d^2 + 2d(T+1)\big) + 2dV$$

$$F_B = 21\,L\,d^2 + 2d(T+1)\,L_{\text{global}} + 4d\,\bar{k}_W\,L_{\text{local}} + 2dV$$

The $$21d²$$ assumes Model B's four query heads per key/value head, so $$n_{kv}·h = d/4$$. With other ratios, the projection
term is $$2d(d + 2·n_{kv}·h) + 2d²$$.

Plugging in the numbers:

- Model A: $$12 × (14.16M + 1.57M) + 77.27M = 266.0M$$ FLOPs per token.
- Model B: $$12 × 12.39M + 3 × 1.57M + 9 × 1.18M + 77.27M = 241.2M$$ FLOPs per token ($$k̄_W = 384.25$$).

### Reproducing this article

The trained models are public on Hugging Face, each with a model card:
[Model A](https://huggingface.co/CuriosLM/gpt2-124m-fineweb-edu), in the standard 🤗 transformers GPT-2 format;
[Model B](https://huggingface.co/CuriosLM/gemma-style-113m-fineweb-edu); and
[Model B on the 50:50 mix](https://huggingface.co/CuriosLM/gemma-style-113m-fineweb-edu-cosmopedia).

Everything else lives in the [CuriosLM repository](https://github.com/Shinibali/CuriosLM):

- **Training:** `train.py` with `config/ladder_gpu.py`. A run is chosen by environment variables:
  - `LADDER_RUNG=a0…a4` picks the rung.
  - `LADDER_BUDGET=9.8B` selects the headline runs.
  - `LADDER_MIX=fineweb_edu:0.5,cosmopedia_v2:0.5` selects the data-mix run.
- **Evaluation:** `evaluate.py` runs lm-evaluation-harness through a small adapter (`curioslm/eval_adapter.py`).
- **Analysis:** one folder per section under `analysis/`. Each folder is a series of numbered scripts that rebuild
  that section's figures from cached data:
  - `ladder/`
  - `headline_comparison/`
  - `step320_spike/`
  - `data_mix/`
  - `efficiency/` (`02_flops.py`, `05_flops_scaling.py`, `06_memory_scaling.py` and `07_mfu.py` for this part; the
    inference scripts are Part 2's)

  `positions/` and `attention/` belong to Part 2.

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

None of the results above came out of a clean first attempt. These are the failures that cost real time or money. In no specific order:

- Resumed runs wasn't exact: A run that stops and restarts should be indistinguishable from one that never stopped. After the first resume, the
loss dropped from 10.2 to 9.7 in a few steps, suspiciously fast. The model was seeing data it had already trained
on. Three separate bugs were behind it:

  - The data loader's position wasn't saved in the checkpoint, so a resumed run started the data over.
  - The batch the loader had already fetched ahead of time was skipped on restart.
  - Evaluation was drawing its batches from the training stream, which shifted where training read next.
  
  The fixes were: save the loader's state with the checkpoint, restore the prefetched batch, and give evaluation
its own fixed batches. Then a test: stop a run, resume it, and check it matches an uninterrupted run bit for bit.

- Local attention made training slower: the first version of rung a4 trained 9% *slower* than the rung before it, because the
window, written as a dense mask, knocked attention off its FlashAttention path, and FlexAttention's block masks won back
about 5%. That fix came with its own lessons. FlexAttention has no backward pass on Apple GPUs or CPUs, so it only runs
on CUDA under `torch.compile`, with the dense mask as the fallback; both paths read the window mask rule from one shared
function, so they can't disagree. On the way there, the mask got built as a tensor instead of a function, flex attention
got applied to the *global* layers too, and a mask buffer was never registered. Enforcing unit tests caught all three.

- Data mix that alternated by step: The 50:50 run blends two datasets micro-batch by micro-batch. The first version of the blending loader decided the
mix the same way on every GPU. On one GPU that's correct. On 8 GPUs, all eight would read FineWeb-Edu on the same
micro-step, then all eight would read Cosmopedia on the next: every *optimizer step* would have been 100% one dataset
or the other, alternating. The fix enumerates the micro-batches globally across GPUs and runs Megatron-LM's blending rule
over that global sequence. Every step is now exactly 50:50 on 1, 2, 4 or 8 GPUs, and the tests check it on each
count. 

- MMLU took three hours to run on Mac M1 pro: Before setting MMLU aside, one run checked whether a model this size could do better than chance on it. That
run crawled on the Mac, while HellaSwag had been fast. The process had grown to 17 GB and was swapping to
disk. MMLU's prompts are long, and scoring needs the full-vocabulary logits: one batch of 16 prompts X 1,024 positions X
50,304 vocabulary entries X 4 bytes is 3.3 GB. The log-softmax made a second copy and the model's own loss
computation a third. HellaSwag's short prompts never got close. A batch size of 4 fixed it, but the real fix is to keep only the rows that are actually scored. The lesson: smoke-test with your *longest* inputs, not the first few.

- GPU Pods init: The first 2-GPU launch on RunPod died before training even started, with
`Failed to bind NVLink SHARP (NVLS) Multicast memory … CUDA error 401`. NVLS is an NCCL feature that some hosts don't
support when you rent a slice of an 8-GPU machine. Some hosts were fine; the 50:50 run's 4-GPU pod hit it again. Setting
`NCCL_NVLS_ENABLE=0` turns the feature off, which costs nothing at 2–4 GPUs and doesn't change the math. 

All of this gets worse with scale: our checkpoints are 12 bytes per parameter (float32 weights plus Adam's two
moment estimates), so a 7B checkpoint is about 84 GB, close to our whole Hugging Face organisation cap of 100 GB. The habits carry over
directly: smoke-test every machine, verify every checkpoint, make the data order replayable.



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

## Citation

```bibtex
@misc{bhattacharyya2026gpt_to_gemma_1,
  author       = {Bhattacharyya, Shinibali},
  title        = {GPT-2 (124M) → Gemma 3-Style (113M), Part 1: Pretraining Lessons from Swapping One Architecture Component at a Time},
  year         = {2026},
  url          = {https://shinibali.github.io/gpt2-to-gemma-part-1/},
}
```
