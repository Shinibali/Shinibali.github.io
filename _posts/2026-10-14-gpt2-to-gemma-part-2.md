---
layout: post
title: "GPT-2 (124M) to Gemma 3-Style (113M) language models, Part 2: KV caches, attention visualization, RoPE clocks and 4× context extension"
date: 2026-10-14
math: true
---

This is the second part of my GPT-2 to Gemma 3-style language model series. In [Part 1](/gpt2-to-gemma-part-1/), I trained a GPT-2 124M (Model A) and a
Gemma 3-style 113M model (Model B) on the same 9.8 billion tokens and found them tied on quality, with the
Gemma recipe slightly slower to train but with a 6.4× smaller KV cache. In this part, I look at the model internals at test time: how
fast they generate text, how they represent position and what they attend to. I also extend Model B's context length
from 1024 to 4096 tokens with YaRN.

Here is the TLDR of what I found:

- On my M1 Pro MacBook, whose memory bandwidth is about 1/17 of an H100's, Model B generates text up to 1.55× faster
  than Model A due to its 6.4× smaller KV cache. On an H100, Model B is 1/3 slower than A. At this size, the H100 spends most of each step on the fixed cost of launching
  GPU operations, and Model B has more of them. So the smaller cache size pays off at serving scale, with big
  batches, long contexts and bigger models.
- RoPE gives Model B 32 fixed "clocks" to read position from: pairs of dimensions that rotate at different speeds as
  the position grows. Model A's learned position table organizes itself into slow waves instead. The two schemes work
  differently, but both models end up encoding position as the distance between tokens.
- Model B's strongest induction head, the circuit that copies text from earlier in the context, sits in a local layer.
  So when the text to copy is, for example, 768 tokens back, Model A still copies it and Model B can't. On ordinary text the two models
  rarely disagree, and B's position scheme even changes where it parks attention it doesn't need.
- YaRN stretched Model B from 1024 to 4096 tokens (4×) with no retraining, taking perplexity at 4096 from 37.7 down to
  23.2. Continuing the training run for 200 steps (1% of pretraining) brings it to 21.0, close to the original model's
  20.6 at 1024 tokens.

## The two models

As a reminder, the two models from Part 1 (124M and 113M parameters, 1024 context length) were trained identically on FineWeb-Edu:

- **Model A** is GPT-2 as nanoGPT ships it: learned position embeddings and full multi-head attention in every layer.
- **Model B** is Gemma 3-style [[9]](#ref-9): RoPE [[1]](#ref-1), grouped-query attention (12 query heads sharing 3 key/value heads), a GeGLU MLP,
  sandwich RMSNorm, and local 512-token attention in three layers out of four.

The two models tied on loss and on every benchmark. On paper, Model B needs fewer FLOPs; in practice, it trained 4% slower. Most of the Gemma recipe targets efficiency and longer contexts. Its two attention changes, GQA and local windows,
shrink the memory a model needs during inference, and RoPE makes position relative and easier to extend to
longer contexts. The sections below test each of these: the cost of generating text, how each model represents
position, and how well Model B's context stretches with YaRN.

## Generating text with a KV cache

Part 1's ladder showed the recipe's main payoff on paper: a 6.4× smaller KV cache. To measure what that buys during
text generation, I benchmarked both models with a KV cache on two machines. The answer turned out to depend more
on the machine than on the model.

### What a KV cache stores

A language model generates one token at a time, and each new token attends to every token before it. Without a cache,
every step reprocesses the whole sequence from scratch, which is wasteful because the keys and values of earlier
tokens never change. A KV cache stores them once and reuses them, so each step only has to process the one new token.

Each layer stores one key and one value (the 2 below) per key/value head and per cached position, so the cache for
one sequence takes

$$\text{KV cache bytes} = 2 \cdot n_{kv} \cdot h \cdot b \cdot \sum_{\ell=1}^{L} T_\ell$$

where $$n_{kv}$$ is the number of key/value heads, $$h = 64$$ the head size, $$b$$ the bytes per number (4 in float32)
and $$T_\ell$$ the number of positions layer $$\ell$$ keeps. Model A keeps all 12 heads for all 1024 positions in all
12 layers: 75.5 MB. Model B keeps only its 3 key/value heads, which its 12 query heads share, a 4× saving. Its local
layers never look further back than their window, so 9 of its 12 layers stop at 511 positions. Together that's
11.8 MB, 6.4× smaller.

### What limits generation speed

Horace He's [*Making Deep Learning Go Brrrr*](https://horace.io/brrr_intro.html) [[12]](#ref-12) sorts GPU time by what the chip is waiting on:

- **Compute:** the chip is busy doing arithmetic.
- **Memory bandwidth:** it's waiting for data to arrive from memory.
- **Overhead:** it's waiting for the next instruction. That covers launching kernels, Python, and moving small
  tensors around.

Which of these dominates decides whether a smaller KV cache makes a model faster.

**Prefill** reads the prompt: every prompt token goes through the model in one forward pass, and their keys and values
are *written* to the cache. With enough tokens that's large matrix multiplications, so prefill leans towards
*compute*, so its speed depends on arithmetic more than on cache size. Model B does 9% less arithmetic per token than Model A (Part 1, "Counting FLOPs").

**Decode** generates the answer one token per step. Each step does very little arithmetic, but it has to read every
weight and every sequence's whole cache. For a large model that makes decode *memory-bound*, which is exactly where a
smaller cache pays off. For a small model there is another cost: each step also pays a fixed cost to launch a few hundred
small kernels, one per operation per layer. If that fixed cost is bigger than the time spent moving data, decode is
*overhead-bound*, and the cache size barely matters.

If you've used an LLM API with "cache write" and "cache read" prices, it's the same split: prefill writes the cache,
decode reads it.

### Mac vs H100

I ran the same benchmark on my M1 Pro MacBook (16-core GPU, 16 GB of unified memory, 200 GB/s memory bandwidth) and on
one H100 (3.35 TB/s), both in float32, taking the median of repeated runs after warm-up. Each generation run produces 256 new tokens.

| | Model A | Model B | B vs A |
|---|---|---|---|
| KV cache per sequence, 1024 tokens (any machine) | 75.5 MB | 11.8 MB | 6.4× smaller |
| **Mac:** generation with cache, batch 8, 512-token prompt | 297 tok/s | 461 tok/s | **1.55×** |
| **Mac:** prefill, batch 8, 1024 tokens | 13.5k tok/s | 12.2k tok/s | 0.90× |
| **H100:** generation with cache, batch 8, 512-token prompt | 1,988 tok/s | 1,290 tok/s | **0.65×** |
| **H100:** prefill, batch 8, 1024 tokens | 212k tok/s | 194k tok/s | 0.92× |

At 1024 tokens, prefill behaves the same on both machines: B is 8–10% slower, in line with Part 1's training slowdown
(on the H100, short single-sequence prompts are even slower, because they are overhead-bound).
Generation is where the two machines differ. On the Mac, B generates up to 1.55× faster (batch 8, 512-token prompt); on the H100, it's a third *slower*.

The time per decode step shows why:

![Decode step time against megabytes read per step, on each machine](/assets/s1/efficiency/figures/decode_regimes.png)

**On the Mac, step time rises with the work done per step.** Each step reads all the weights plus the whole cache,
between 457 and 875 MB across the eight settings, and the bytes read alone explain 84% of the variation in step time.
More sequences and longer prompts mean more cache to read, and B reads less of it, so B wins in all but the smallest setting, and by more as
the batch grows.

**On the H100, step time is flat.** Model A takes 3.8–4.0 ms per step and Model B 6.1–6.3 ms, whether it's generating
for one sequence or eight, from a 128-token prompt or a 512-token one. Reading 460–880 MB at the H100's 3.35 TB/s
takes 0.14–0.26 ms, a tiny part of each step. The rest is overhead: a few
hundred kernel launches per step, each costing microseconds, adding up to milliseconds. Model B launches more of them,
with two extra normalizations per block, the gating multiply, the rotations and the local-window masks. So on the
H100 it's slower, and its smaller cache never gets a chance to matter.

![Generation speed with and without the cache, and prefill speed, on the Mac](/assets/s1/efficiency/figures/inference_speed.png)

The same applies to the speed-up from the cache itself. On the Mac it speeds up generation 2.5–5.3× at batch 1 and 8–23× at batch 8,
because recomputing the whole sequence every step is a lot of extra work. On the H100, at batch 1 with a short prompt,
the cache gives no speed-up at all (0.9–1.0×), because recomputing 128 tokens costs no more launch overhead than
processing one. Its benefit only appears with longer prompts and bigger batches (up to 6.3× for Model A at batch 8).

![KV-cache memory as the context grows](/assets/s1/efficiency/figures/kv_cache.png)

The memory plot shows GQA's saving and the windows' saving separately. GQA divides the slope by four. The local windows make most of the
layers stop growing at 511 positions, so past that point only B's three global layers keep adding to the cache. The
measured sizes match the formula exactly, on both machines.

### When the smaller cache starts to pay off

So when would the H100 result flip? For cache traffic to outweigh about 4 ms of overhead, a step has to
read around 13 GB, roughly 175 full-length sequences of Model A's float32 cache at once. Real serving systems work at
that scale: large batches, long contexts, bigger models, and CUDA graphs and kernel fusion that cut the per-step
overhead. A 124M model on a single H100 isn't there, so small-scale inference benchmarks, including this one, should be
read with that in mind.

### The KV cache of a 7B model

Arithmetic intensity, the FLOPs done per byte read, tells which part of decoding can ever become compute-bound. For the
weights, each parameter read in bf16 (2 bytes) is used for 2 FLOPs per sequence in the batch, so the intensity is
about the batch size. An H100 does about 295 bf16 FLOPs per byte it can read, so decode only becomes compute-bound
for the weights at batches of a few hundred sequences. Attention over the cache is different. Each cached key and
value is read for one sequence only, and is used by every query head that shares it: $$n_h / n_{kv}$$ heads, which is 1 in Model A and 4 in Model B.
That gives an intensity of 1 FLOP per byte for Model A and 4 for Model B, *whatever the batch size*. Reading the cache is memory-bound at any scale,
and GQA's real job is to multiply that intensity by the number of query heads per key/value head.

The same formula gives the KV cache per sequence of a 7B-class model (32 layers, $$d_{model}$$ = 4096) in
bf16:

| Context | Full multi-head attention | GQA (8 KV heads) | GQA + Gemma 3-style local windows |
|---|---|---|---|
| 8k tokens | 4.3 GB | 1.1 GB | 0.31 GB |
| 32k tokens | 17.2 GB | 4.3 GB | 0.91 GB |
| 128k tokens | 68.7 GB | 17.2 GB | 3.3 GB |

At 32k tokens, one sequence's full-attention cache would fill a fifth of an 80 GB H100. The Gemma 3-style layout fits
nearly twenty sequences in the same space. Overhead stops dominating too: reading a 7B model's bf16 weights alone
takes about 4 ms per decode step at 3.35 TB/s, as long as the whole fixed overhead measured here. That's the regime
the Gemma design targets, where a 6.4× smaller cache means 6.4× more sequences in the same memory. At that scale the Mac
result, where the smaller cache made decoding faster, is the one that carries over.

## FLOPs at long contexts

Part 1 counted each model's FLOPs per token, term by term. Attention is the only term that grows
with context length, so the same algebra tells us what long contexts do to a model.

Per layer, attention costs $$2d(T + 1)$$ FLOPs per token against the MLP's $$16d^2$$, so **attention overtakes the MLP when $$T ≈ 8d$$**, with $$d$$ the model dimension ($$d_{model}$$). For our $$d_{model}$$ of 768 that's about 6,000 tokens; for a 7B-class
model ($$d_{model}$$ = 4096), about 33,000. A local layer's queries see at most $$W$$ keys, so its attention cost levels off at $$4d·W$$ per token. That only reaches the MLP's $$16d^2$$ if $$W > 4d$$ (about 3,000 tokens at our $$d_{model}$$), so with Model B's
512-token window the local layers never cross over; only the global layers do.

To see this in practice, I scaled each recipe from Part 1 up to a 7B-class and a 14B-class
model and counted the forward FLOPs per token at a 1k and a 128k context. The numbers below are absolute FLOPs per token:

![Forward GFLOPs per token for Model A's and Model B's recipes, at two sizes and two context lengths](/assets/s1/efficiency/figures/flops_by_context.png)

| Forward GFLOPs per token | 1k context | 128k context | Attention scores' share at 128k |
|---|---|---|---|
| Model A's recipe, 6.7B | 13.6 | 47.7 (3.5×) | 72% |
| Model A's recipe, 15.4B | 31.2 | 95.1 (3.0×) | 68% |
| Model B's recipe, 5.8B | 11.9 | 20.5 (1.7×) | 43% |
| Model B's recipe, 13.5B | 27.3 | 43.4 (1.6×) | 38% |

With full attention, stretching the context from 1k to 128k tokens multiplies the work per token by 3–3.5×, almost
all of it attention scores. With Model B's recipe (512-token windows in three layers of four), the same stretch
costs 1.6–1.7×, because only the global layers pay for the long context. So at 128k tokens, a 13.5B model
built with Model B's recipe does *less* work per token than a 6.7B model built with GPT-2's (43 against 48 GFLOPs).
That's on top of the memory savings from the KV cache section above. Gemma 3's report [[9]](#ref-9) gives memory as its main reason for going to five
local layers in six at 128k, but the compute arithmetic points the same way.

The same counting matters for how training efficiency is reported. The PaLM formula behind Part 1's logged MFU
charges every token for attending to the whole context; a causal layer computes about half of that, and a local
layer only its window. At 1024 tokens, for the 7B-class models above, the formula overstates the work by 2–3%. At
128k tokens it overstates it by 72% for Model A's recipe and by a factor of 3.9 for Model B's, because it charges
the local layers for the full 128k they never look at. A windowed long-context model logging "40% MFU" this way is
really using about 10% of its GPUs. So an MFU number for a long-context or hybrid-attention run is only meaningful if
you know which FLOP count it assumes.

### A note on attention's quadratic cost

Per token, attention is linear in context length; summed over a whole sequence it's the familiar quadratic,
about $$2d·T^2$$ FLOPs per layer. FlashAttention (Dao et al., 2022) [[11]](#ref-11) does the same arithmetic but cuts memory
traffic, so the $$T × T$$ score matrix never has to sit in GPU memory, which is what makes long-context training fit
at all. Within exact attention, only restricting which keys a query sees, as Model B's local windows do, changes the order, to $$O(T·W)$$
per sequence, and only with a kernel that skips the masked blocks. The appendix
([Attention and FlashAttention](#attention-and-flashattention)) works through
the numbers.

## How each model represents position

The two models end with the same loss, but they use very different machinery for the one thing a
transformer can't know by itself: where each token sits in the sequence. Model A learns a vector for every position
from scratch. Model B is told position through a fixed rotation. Below, I look at what each one actually learns.

### RoPE as 32 clocks

RoPE [[1]](#ref-1) splits each 64-dimensional attention head into 32 pairs of dimensions and rotates each pair by an angle
proportional to the token's position. Each pair turns at its own speed. The fastest completes a full turn every 6
positions. The slowest would need 47,000 positions to complete one. When a query and a key are compared, only the
*difference* of their angles survives, so attention depends only on how far apart two tokens are.
In effect, each head reads the distance between two tokens off a set of 32 clocks running at different speeds.

![Wavelength of each RoPE band against the 1024-token training context](/assets/s1/positions/figures/rope_bands.png)

Only 18 of the 32 clocks complete a full turn within the 1024-token training context. The other 14 never do: during
training, the model only ever sees a fraction of their cycle. That's harmless at 1024 tokens, but it becomes the
problem when the model reads longer inputs, and it's what YaRN fixes later in this post.

### How an attention head uses the clocks

To see what these rotations do to attention, it helps to treat each pair of dimensions as one complex number. I use $$j$$ for the band index below, to keep it apart from the imaginary unit $$\mathrm{i}$$.

Take one head, with head size $$h = 64$$, and band $$j$$: dimensions $$j$$ and $$j + h/2$$, which RoPE rotates by $$m\theta_j$$ at position $$m$$, where $$\theta_j$$ is the band's fixed speed. The key at position $$n$$ is rotated the same way, by $$n\theta_j$$.

**Step 1: a rotation is a multiplication.** The RoFormer paper [[1]](#ref-1) uses the same complex form. Write the query's pair as one complex number

$$z_j^q(m) = q_j(m) + \mathrm{i}\,q_{j+h/2}(m),$$

and the key's as $$z_j^k(n)$$. Rotating a 2-D vector by an angle $$\alpha$$ is the same as
multiplying its complex number by $$e^{\mathrm{i}\alpha}$$. So RoPE turns the query's pair into $$z_j^q(m)\,e^{\mathrm{i} m \theta_j}$$, and the key's into $$z_j^k(n)\,e^{\mathrm{i} n \theta_j}$$.

**Step 2: a dot product is the real part of a product.** For two 2-D vectors written as complex numbers $$a$$ and $$b$$,
their dot product is $$\mathrm{Re}[a\,\bar{b}]$$, where $$\bar{b}$$ is the complex conjugate. So band $$j$$'s share of the attention score is

$$\frac{1}{\sqrt{h}}\,\mathrm{Re}\Big[\, z_j^q(m)\,e^{\mathrm{i} m \theta_j} \cdot \overline{z_j^k(n)\,e^{\mathrm{i} n \theta_j}} \,\Big]
= \frac{1}{\sqrt{h}}\,\mathrm{Re}\Big[\, z_j^q(m)\,\overline{z_j^k(n)} \; e^{\,\mathrm{i} (m-n)\,\theta_j} \Big]$$

because the conjugate of $$e^{\mathrm{i} n\theta_j}$$ is $$e^{-\mathrm{i} n\theta_j}$$, so the two rotations combine into one. **Only the difference $$\Delta = m - n$$ survives.** This is what makes RoPE relative.

**Step 3: separate content from position.** The product $$c_j = z_j^q(m)\,\overline{z_j^k(n)}$$ depends only on what the
query and key vectors contain before rotation. Write it as a size and an angle, $$c_j = |c_j|\,e^{\mathrm{i}\varphi_j}$$. Summing
over the 32 bands gives the whole score:

$$\text{score}(\Delta) = \frac{1}{\sqrt{h}} \sum_{j=1}^{32} |c_j| \cos\!\big(\Delta\,\theta_j + \varphi_j\big)$$

In words: the attention score, as a function of the distance $$\Delta$$ between two tokens, is a sum
of 32 waves. **The waves' frequencies are fixed by RoPE. How much of each wave a head uses, and where each wave
peaks, are set by content**, through the query and key weights the head has learned. So each head builds its own
preference for distance by mixing fixed waves, a bit like an equaliser setting the volume of fixed
frequency bands.

So a head can choose how it treats distance. Fast waves give a sharp preference for nearby
tokens. Middle waves give a smooth fade over tens to hundreds of tokens. Slow waves, the ones that never complete a turn within the context,
give almost no preference at all, so a head built on them decides mostly based on *what* the tokens are.

Do Model B's trained heads actually split like this? To check, I grouped the bands the way the YaRN paper [[3]](#ref-3) does,
by counting how many full turns each band makes within the 1024-token training context:

- **Fast bands** make 32 turns or more (6 of the 32 bands).
- **Middle bands** make between 1 and 32 turns (12 bands).
- **Slow bands** never complete a single turn (the other 14).

These are the same groups YaRN treats differently in the YaRN section below: it leaves the fast bands alone, rescales the slow
ones, and blends the middle. The thresholds are counted in turns, so they mean the same thing at any context length,
but which bands they select shifts with the context. (The 1-turn cut-off follows from the definition; the 32-turn cut-off is an
empirical choice the YaRN authors made for Llama models.)

The first figure answers this from the trained weights alone. A head can only use a band as much as its query and
key projections feed that band's pair of dimensions. The head's **q·k weight** in band $$j$$ is the size of the query and
key projection rows that feed band $$j$$, multiplied together, as a share of the head's total. Its **slow-band weight** sums
that share over the 14 slow bands; a head that treated every band alike would get 14/32, or 44%. (The formula is in the
appendix, [How the band measures are computed](#how-the-band-measures-are-computed).)

![Which RoPE bands each of Model B's 144 heads uses, measured from the trained weights](/assets/s1/positions/figures/band_usage.png)

Each row is one head (12 per layer, 12 layers) and each column one band, fastest on the left. Darker means more of
that head's q·k weight. The dashed line separates the bands that turn within the context from the slow ones. The
pattern leans toward the slow bands on the right:

- **Overall:** the median slow-band weight is 53%, against 44% for a head that treated every band alike.
- **Five heads have a slow-band weight above 75%:** one in layer 1, three in layer 6 and one in layer 8. They're the
  dark stripes on the right.
- **No head concentrates on the fast bands.** The lowest slow-band weights, mostly in layer 0, sit at or just below the 44%
  of an even spread (lowest 42%).

This matches what Barbero et al. (2025) [[2]](#ref-2) found in a much larger Gemma model: a strong preference for the lowest
frequencies, which they read as carrying meaning, and the highest frequencies used for positional patterns.

The weights only show what a head *can* use. The second figure shows what it *does* use on real text. It runs held-out
FineWeb-Edu through Model B and, for three heads, averages the pre-softmax score $$q·k/\sqrt{h}$$ over every query–key pair the
same distance apart, so higher means the head prefers keys at that distance.

Using the complex form from the derivation above, that average score splits exactly into one contribution per
band (formula in the appendix). The figure's bottom row is the head's mean score at each distance, and the top
row sums the bands' contributions within each speed group: fast, middle and slow.

![Three real Model B heads: average attention score vs distance, and the contribution of fast, middle and slow bands](/assets/s1/positions/figures/real_heads.png)

**How to read it.** Each column is one real head, chosen from the heatmap:

- the highest slow-band weight (layer 6, head 11, 84%);
- the median (layer 9, head 1, 53%);
- the lowest (layer 0, head 6, 42%).

Panel titles mark global or local layers; local heads' curves stop at $$W = 512$$. The bottom row is the head's average
score at each distance, and the top row splits it into the parts from its fast, middle and slow bands. The three parts
add up exactly to the bottom row.

Each head matches one of the behaviours described earlier:

- **Layer 0, head 6 is sharply local.** Its score halves from distance 0 to distance 1 and keeps falling. That first
  drop is spread across all three speed groups.
- **Layer 9, head 1 fades smoothly** over about 100 tokens, through its middle and slow bands together.
- **Layer 6, head 11 barely cares about distance** beyond its immediate neighbours: its score is nearly flat from 3
  tokens out to the edge of its 512-token window, and its fast bands contribute nothing at all.

One caution when reading these curves: they mix two things, RoPE's rotations and the fact that nearby tokens tend to be
similar in content. The steep drop over the first few tokens in the slow bands, in layer 6's head and layer 0's, can't
come from rotation, since those bands barely turn in a few tokens. It's content: a query matching keys from its own
neighbourhood. The flat stretch beyond that is where the slow bands' indifference to position shows.

### Model A's learned position table

Model A's position table is just 1024 free vectors; nothing in its design says anything about distance, smoothness
or waves. Still, training gives it structure. Model B has no such table, since RoPE rotates queries and keys instead
of adding a vector to each token, but one thing can be compared directly: how similar two positions look as a
function of their distance.

![Position as waves: Model A's learned table, compared with Model B's RoPE](/assets/s1/positions/figures/learned_positions.png)

**Left: Model A's position vectors, every one compared with every other.** The bright diagonal band says nearby
positions have similar vectors, and the similarity fades with distance. Distance alone explains 71% of the variation
in this matrix. The model has learned something close to "position as distance", the very property RoPE builds in. Wang & Chen (2020)
[[16]](#ref-16) found the same kind of structure in the learned position embeddings of pretrained models, GPT-2 among them.

**Middle: the three directions along which Model A's position vectors vary most**, plotted against position.
They're smooth waves: half a cycle, a full cycle, about one and a half. This is expected: when similarity
depends only on distance, the matrix's eigenvectors are close to sinusoids, the same mathematics as a Fourier transform.

**Right: similarity against distance, for both models.** For RoPE, rotate any vector to two positions and
compare: the result is a sum of cosines of the distance, one per band, so it depends on distance alone by
construction. (The curve gives every band equal weight, which is position alone, before a head's content decides
which bands matter.) The two curves have opposite shapes:

- **RoPE is sharp up close and blurry far away.** Its similarity falls from 1.0 to 0.66 within 10 positions, as the
  fast bands lose alignment, and then drifts slowly down, to about 0.2 by the end of the window, held up by the slow bands that barely turn. Nearby
  positions are easy to tell apart, while distances of 600 and 900 tokens look much alike.
- **Model A's table is blurry up close and coarse-grained far away.** Its similarity is still 0.91 at distance 10.
  Neighbouring positions get nearly the same vector, but by 500 positions apart the similarity turns negative: the
  table mostly encodes *roughly where in the window* a token sits.

The RoPE curve is jagged because it's exact, the sum of the 32 cosines at each whole-number distance; Model A's is
smooth because it averages every pair of positions at that distance. Model A's shaded band, the middle 80% of those
pairs, shows that its similarity also depends on *where* in the window a pair sits; RoPE has no band, because every
pair at the same distance gets exactly the same value.

Breaking Model A's table into waves along the position axis (a Fourier transform) shows the same thing: two-thirds of its
variation sits in a handful of slow waves, 341 positions and longer, and most of the rest, about 22%, is a flat,
noise-like floor at short wavelengths. RoPE puts structured waves exactly there, from 6 positions up, and keeps 14
bands slower than the whole window, which a table of 1024 rows can't represent at all.

That's one plausible reason RoPE was the one change in Part 1's ladder that improved loss, and it's why YaRN, later in this
post, has slow bands to rescale.

### Position at longer contexts

At longer contexts, the two schemes behave very differently. Model A's position table simply stops at 1024 rows, so it
can't read further at all. Building one for a 128k context in a 7B-class model would take 537M parameters, each row
trained only on the tokens that happen to land at its position. This is a big part of why learned position
tables have all but disappeared from long-context models.

RoPE needs adjusting too. With the base of 10,000 used here, every one of a 128-dimensional head's 64 bands completes
a turn within 128k tokens, so no band is slow enough to tell "far" from "very far". Long-context models raise the
base: Llama 3 [[10]](#ref-10) uses 500,000, and Gemma 3 uses 1,000,000 on its global layers. That leaves 15 and 17 bands slower than
the whole 128k context. Model B's windows save more and more memory as the context grows, but more and more of the
context also lies outside them. In Gemma 3's layout, five layers in six never look beyond 1024 tokens, even at a
128k context, so everything long-range has to travel through the global layers. That's the price of the memory savings, and the reason
the design keeps some global layers.

## Visualizing attention

This section looks at what each model attends to while it reads. A word of caution first: attention weights show where
information *can* flow, which is weaker evidence than showing why a model made a prediction (Jain & Wallace, 2019) [[13]](#ref-13). And a weight averaged over a
layer's heads can hide a single head doing something sharp. So I treat the views below as a way to look inside the models.

A small example first, with emojis! 🤗

GPT-2's tokenizer works on bytes, and
an emoji is four bytes of UTF-8. In ***"Learning is Fun 🤗, Keep exploring 🔭"***, each emoji becomes three tokens: one they
share (a space and the first two bytes, which every emoji from U+1F000 to U+1FFFF begins with) and two of their own.
Reading the shared token, a model can't know which emoji is coming; only the later tokens decide. Both models make
the same bet there. At the shared first token of the telescope, Model A gives 86% to the next byte of the hugging face
and Model B 68%: having seen one emoji, each expects to see it again.

Asked to continue the line, always picking the most likely next token, both models copy it:

| | Greedy continuation of "Learning is Fun 🤗, Keep exploring 🔭" |
|---|---|
| Model A | ↵ Learning is Fun 🤗, Keep exploring **🤗** ↵ Learning is Fun 🤗, Keep exploring **🤗** ↵ … |
| Model B | ↵ Learning is Fun 🤗, Keep exploring **🔭** ↵ Learning is Fun 🤗, Keep exploring **🔭** ↵ … |

Model B copies the line exactly. Model A copies it with the wrong emoji at the end, every time: at the token the two
emoji share, it completes the first one it saw. Small models picking greedily fall into loops like this; that's a
known property of greedy decoding (Holtzman et al., 2019) [[8]](#ref-8). With sampling instead, both write ordinary on-topic text: Model A continues "for more learning! 🔨 This is
the first part of our series of lessons!", Model B "for more fun learning activities. 1. The Elements of Play".

The copying is visible in the attention itself. For each model, the figure below shows every layer's attention,
averaged over its heads, on the prompt and the first copied line.

![Model A: token-to-token attention in every layer, on the probe and its first copied line](/assets/s1/attention/figures/emoji_tokens_a.png)

![Model B: token-to-token attention in every layer, on the probe and its first copied line](/assets/s1/attention/figures/emoji_tokens_b.png)

**How to read it.** Each panel is one layer. Rows are the token doing the looking, columns the tokens it looks at;
darker means more attention, and the grey triangle is the future, which a causal model never sees. The top half of
each panel is the prompt; the bottom half is the copied line. A copying layer shows a stripe running parallel to the
diagonal in the bottom half: each token of the copy looks back at the token right after its twin in the first line,
which is the next thing to write.

The copy stripe is strongest in one layer per model (with a weaker echo in layer 8): layer 7 in Model A and layer 6 in Model B,
the same layers that hold each model's strongest induction head (see below). The other layers do simpler things. Model B's layer 0 mostly attends to the
token itself and its layer 2 to the previous token, the kind of local bookkeeping a copying head needs as an input.
Both models also park a lot of attention on the opening `<|endoftext|>` token, Model A especially in its later
layers.

A single short sentence never reaches Model B's 512-token windows, so for the rest of this
section I measure every head on full-length text and test one mechanism, copying, directly.

### Attention distance and attention sinks

![Mean attention distance and first-token attention for every head of both models](/assets/s1/attention/figures/attention_distance.png)

**How to read it.** Each dot is one head, the line its layer's average; blue is Model A, orange Model B, the shaded
columns Model B's three global layers, and the dashed line Model B's window. The left panel is *attention distance* [[7]](#ref-7):
how many tokens back a head looks on average, weighting each earlier token by its attention (a measure from the
Vision Transformer paper). It uses only queries in the second half of 32 held-out 1024-token windows, so every head could look at
least 512 tokens back. The other panels measure *attention sinks* [[6]](#ref-6), a fixed place models park
attention when nothing else needs it: the share on the window's first position, whatever token sits there (middle),
and on document separators, the token that starts every page (right).

In symbols, with $$p_{q,k}$$ the attention a head puts from the query at position $$q$$ on the key at position $$k \le q$$
(each query's weights add up to 1), each panel averages one of these over the queries:

$$\text{distance}_q = \sum_{k \le q} p_{q,k}\,(q - k) \qquad \text{first-position share}_q = p_{q,0} \qquad \text{separator share}_q = \sum_{k \le q,\ x_k = \texttt{EOT}} p_{q,k}$$

The separator share is averaged only over the windows that contain a separator.

Model A's heads look further back on average: 132 tokens against Model B's 81. A has three layers averaging more
than 200 tokens (0, 7 and 10) and single heads reaching almost 480. B's global layers don't compensate: its most
far-sighted layer, global layer 8, averages 148 tokens, and its farthest-looking head 235. On ordinary text, B simply
looks less far, and none of its heads comes close to the 512-token window that would stop it.

The two sink panels show the clearest difference between the recipes. Model A rests attention on position 0 (up to
13% of a layer's attention, in its later layers) and even more on document separators (up to 29%). Model B's global layers never
rest on position 0 (its local layers can't reach it from these queries), but B does rest on separators, though less than A (at most 11% of a layer, in layers 6, 8 and 10). BERT's heads
behave similarly with its [SEP] separator token, which Clark et al. (2019) [[15]](#ref-15) read as a way for a head to do
nothing. The likely reason for the difference between A and B is how each model knows where it is. A's learned position table gives position 0 its own vector, so the first
position is recognisable whatever token sits there. RoPE gives position 0 no vector of its own, so B has nothing built in that marks it, and it rests on a
token it can recognise instead.

### Induction heads and the local window

The best-documented circuit in language models is the *induction head*, first described by Elhage et al. (2021)
[[14]](#ref-14) and studied in depth by Olsson et al. (2022) [[5]](#ref-5): at a token it has seen before, it attends to
the token that followed the earlier occurrence, so the model can copy what came next. The standard test, from those papers, removes every
other clue: a block of 64 random tokens, random filler, then the same 64 tokens again. Nothing can predict a random
token except the earlier copy. Varying the filler puts the repeat either 256 tokens back, inside Model B's local
window, or 768 tokens back, beyond it. Model A can attend 768 tokens back in any layer. Model B can only do it in its
three global layers.

![Each model's strongest induction head on a short repeated block of random tokens](/assets/s1/attention/figures/induction_heads.png)

**How to read it.** Same format as the emoji figure, for each model's strongest induction head reading 12 random
tokens and then the same 12 again. In the first copy the head rests on the first token; in the second, each token
attends to the token right after its earlier twin, the stripe one step right of the earlier copy.

![Induction score of every head at two repeat distances, and each model's loss on the repeated tokens](/assets/s1/attention/figures/induction.png)

**How to read it.** In the two left panels, each dot is one head, and its *induction score* is the attention it puts
on exactly the token it should copy, averaged over the repeated block: near 1 is a clean induction head, near 0 is
none. The shaded columns are Model B's global layers again. The right panel asks whether the copying works: the loss
on the second copy of the block, against the dashed line for the first copy, which neither model has any way to
predict. A loss near zero means the model copies.

In symbols: if the block's first copy starts at position $$s$$ and its repeat sits $$D$$ tokens later (256 or 768), then
the token at position $$t$$ in the second copy has its twin at $$t - D$$, and the token to copy is the one right after
it, at $$t - D + 1$$. A head's induction score averages its attention there over the $$n - 1$$ tokens of the second copy
that have a next token to copy ($$n = 64$$), and over the test sequences:

$$\text{induction score} = \frac{1}{n-1}\sum_{t=s+D}^{s+D+n-2} p_{t,\;t-D+1}$$

The loss on each copy is Part 1's cross-entropy, $$-\ln p(x_{t+1} \mid x_{\le t})$$, averaged over the copy's tokens.

With the repeat 256 tokens back, both models have clean induction heads: Model A's strongest is layer 7, head 11
(score 0.93), with 5 heads above 0.3; Model B's is layer 6, head 11 (0.90), with 4. Both copy, and Model A copies much better.
The second copy of the block costs Model A 1.6 nats per token and Model B 6.1, against about 13.3 for the unseen
first copy.

With the repeat 768 tokens back, Model A is barely affected: the same head still scores 0.90, and the second copy costs
3.0 nats. Model B's induction heads are gone. No head scores above 0.3, its best manages 0.05, and the second copy
costs 12.3 nats, almost as much as tokens it has never seen. Its strongest induction head lives in layer 6, a local
layer, and the earlier copy is now out of that layer's reach. None of its three global layers took over the job.

This is a cost of the local/global recipe that average loss doesn't show. Copying from far back is a narrow
skill, but a real one, and in Model B the windows' memory savings came at exactly that cost.

### Where the two models disagree

Average loss can hide where two models differ, so I compared their predictions token by token on held-out text,
using the KL divergence between the two next-token distributions, KL(A ‖ B): the extra surprise, in nats, of
predicting with Model B's distribution when Model A's is the reference. Zero means identical predictions.

For one position, summing over every token $$v$$ in the vocabulary:

$$\mathrm{KL}(A \,\|\, B) = \sum_{v} p_A(v)\,\big(\ln p_A(v) - \ln p_B(v)\big)$$

![Per-token KL between the two models, and each model's attention at the strongest disagreement](/assets/s1/attention/figures/disagreement.png)

**How to read it.** The left panel counts tokens by how much the models disagree, on a log scale: most tokens sit in
the middle, a few in the long tail on the right. The two heatmaps take the single strongest disagreement and show,
for each model and layer, where its attention went from that token, averaged over heads. The first column collects the
attention that went to tokens before the passage shown, so each row adds up to 1.

Over 20,480 held-out tokens, the typical disagreement is small: a mean KL of 0.29 nats and a median of 0.21. Only 3%
of tokens exceed 1 nat and 0.1% exceed 3. On these 20,480 tokens, average loss tells Part 1's story: 2.98 for A, 2.97 for B.

In the tail, B does slightly better: among the 1,000 strongest disagreements, B
has the lower loss on 59%. The strongest few are mostly completions where one model knows the phrase and the other
doesn't:

| Context (end) | True next token | Model A's guess | Model B's guess |
|---|---|---|---|
| "…capable of supporting up to 127 daisy" | "-" (daisy-chained) | "ph" 48% | **"-" 59%** |
| "…whether individuals use their dominant versus their" | " nond(ominant)" | " dominant" 36% | **" nond" 99%** |
| "…the measurement of these inflammatory 'medi" | "ators" | "as" 63% | **"ators" 87%** |
| "…if you have 40 CALs … The 41st" | " user" | " mode" 12% | " CAL" 96% |

The last row goes the other way: Model B is very sure of a word that's wrong. At the strongest disagreement,
"daisy-chained", the attention is spread in a similar way in both models: about half or more goes to tokens before the passage (60% in A,
48% in B), a fifth to a quarter to "da" and "isy", the word being completed, and the rest across the passage. So the two models look at similar places and
draw different conclusions from them, which attention weights alone can't show.

## Extending Model B's context from 1024 to 4096 with YaRN

Model A can never read past 1024 tokens: its position table has no row 1025. Model B has no table, so in
principle RoPE can rotate a query to any position. In practice, a RoPE model run past its training length falls apart
quickly. The RoPE clocks above show why: the slow bands reach angles never seen in training, and a head's mix of waves was only
tuned on distances up to 1024.

[YaRN](https://arxiv.org/abs/2309.00071) is a recipe for fixing this after training. It leaves the fast bands alone,
since they already cycled many times and their behaviour at new positions is familiar. It stretches the slow bands so
that 4096 positions map into the range of angles the model saw at 1024, blends smoothly between the two in the
middle, and adds a small correction to the attention temperature. The paper shows it working with no training at
all, and better with a short fine-tune on long sequences.

In terms of the fast, middle and slow bands from earlier, YaRN's default thresholds leave Model B's 6 fast bands alone, fully interpolate the 14 slow
ones and blend the 12 in between, so 26 of the 32 bands in every head get rescaled. Stretching the
context 2×, 4× or 8× without it would make 3, 5 or 7 more bands complete their first full turn, at angles the model
never saw in training.

![Each RoPE band's wavelength in Model B, as trained and with YaRN at 4×](/assets/s1/positions/figures/yarn_bands.png)

**How to read it.** Each dot is one of Model B's 32 RoPE bands, fastest on the left. Hollow dots are the wavelengths
the model was trained with, filled dots the wavelengths YaRN gives it, and the vertical line joining them is how far
YaRN moved that band. The two dashed lines mark the training length (1024) and the new target (4096). Purple bands
don't move. Gold bands are stretched the full 4×, and every one of them lands above the 4096 line. Teal bands are
blended, moving less the faster they are.

One number sums up what YaRN does here. Run plainly at 4096 tokens, 23 bands would complete a full
turn, 5 more than in training. With YaRN it's 18 again: the same bands that turned within 1024 tokens during
training are the only ones that turn within 4096 now. Every band sees a range of angles it already knows.

**Why 4×.** The YaRN paper's [[3]](#ref-3) zero-shot results hold up well to about twice the training length and
degrade beyond it; the long-context results it's known for come after a short fine-tune. At 2×, zero-shot YaRN should
already be close, leaving the fine-tune little to show; at 8×, the target is beyond zero-shot range and pages that
long are rare (about 1% of FineWeb-Edu validation pages reach 8k tokens, against about 3% for 4k). At 4×, 5 bands
newly complete a turn and there's room for the fine-tune to close the gap. Both versions are measured at 4096
tokens.

### Design choices the paper leaves open

**Which layers to stretch.** Three of Model B's four layers use a 512-token local window, so they never look more
than 511 tokens back at any context length; only the global layers ever meet a distance they weren't trained on.
Released models handle this differently. Gemma 3 stretches only its global layers (and with linear
interpolation rather than YaRN). [Rnj-1.5](https://huggingface.co/EssentialAI/rnj-1.5-instruct) and gpt-oss, both
mixing local and global layers, apply YaRN to every layer alike. Since there's no settled convention, I followed the paper,
Rnj-1.5 and gpt-oss: one stretched table for every layer. YaRN leaves the fast bands, which do most of a local layer's
work, untouched anyway.

**The attention temperature.** YaRN also multiplies every query and key by $$0.1\ln s + 1$$, which is 1.139 at $$s = 4$$,
so attention scores grow by $$1.139^2 ≈ 1.30$$ and attention gets slightly sharper. It compensates for the squeezed slow bands
and for having four times as many keys to spread attention over, both of which flatten attention at the new length.
Rnj-1.5's released config turns this correction off, presumably because a long mid-training stage let the model find
its own sharpness. Model B keeps it: zero-shot there is no training to absorb the change, and keeping the same setting
in both versions means the fine-tune is the only difference between them. Folding the factor into RoPE's cosine and
sine tables applies it to both q and k for free, which is why the paper writes it as a square root.

### YaRN without any training

The test set is 500 held-out FineWeb-Edu pages, each at least 4096 tokens long, each read from its start in one pass.
Every window is therefore one real page, and position $$p$$ always has exactly $$p$$ earlier tokens of the same page to draw
on. With causal attention, one pass at 4096 tokens also gives the answer for every shorter length, since a token's
loss never depends on what comes after it.

![Perplexity against length, and loss at each position, for Model A, Model B as trained, and Model B with YaRN with and without fine-tuning](/assets/s1/positions/figures/yarn_context.png)

**How to read it.** The left panel is the perplexity of a page's first $$L$$ tokens, the way the YaRN paper reports it;
the right panel is the average loss at each position. The dashed line is the training length. Model A's line stops
there because it has no position past 1024 to look up. The teal line is the fine-tuned model, which the next part
of this section introduces.

Run past its training length, plain RoPE breaks down. Its loss at each position jumps from about 3.0 to 4.2 between
1024 and 4096, and the perplexity of the whole 4096-token page climbs to 37.7. With YaRN and no training at all, it
stays at 23.2: 14.6 lower, with a 95% interval of [14.1, 15.1] from a paired bootstrap over the pages. So YaRN removes
the breakdown.

But it has a cost. At 1024 tokens, where the original model needed no help, YaRN costs perplexity 20.6 → 22.4: the
sharper attention and squeezed slow bands apply at every length, including the short ones. The right
panel shows what happens past the training length: zero-shot YaRN's loss holds level at about where it was before 1024 (3.15
at positions 2048–4096, against 3.13 at 512–1024), but at that slightly worse level throughout. So the model is
stable at 4096 tokens, with no gain yet from the extra context.

### Fine-tuning setup

The paper fine-tuned Llama 2 7B for 400 steps of 64 × 64k-token sequences from long books (about 1.7B tokens,
roughly 0.1% of its pretraining). Those numbers don't translate cleanly to a 124M model trained on 9.8B tokens.
Matching the *share* of pretraining gives about 10M tokens, about 20 of Model B's steps. Matching the *steps and
batch* gives about 105M. There is no scaling law for how much a YaRN fine-tune needs, and the paper's learning rate
(2e-5) says little about what a 124M model needs.

So the fine-tune continues Model B's own training run and changes as little as
possible:

- It starts from the final checkpoint at step 20,000 with the optimizer's state restored, so AdamW's running averages
  carry over.
- The learning rate stays where the cosine schedule ended, at 6e-5, with no new warmup. A run resumed mid-schedule
  doesn't re-warm, and the restored optimizer state already matches the model.
- Each step is still 491,520 tokens, with each micro-batch now 15 sequences of 4096 tokens instead of 60 of 1024. Weight decay, gradient
  clipping, precision and seed are unchanged.

What does change is the sequence length, YaRN being switched on, and the data. The pretraining stream packs pages back to back,
so most 4096-token windows would join several short pages and seldom reward looking far back. The fine-tune reads
only pages at least 4096 tokens long, one page per sequence, like the paper's long books: 3% of pages but about a
quarter of the tokens. The model has seen these pages once already: pretraining made
one pass over almost all of the 10B-token sample. Repeating about 1% of the data once is well within what
[Muennighoff et al. (2023)](https://arxiv.org/abs/2305.16264) [[4]](#ref-4) found to be as good as fresh data. The evaluation pages
come from the held-out split, which the model never trained on.

The run is 200 steps, about 98M tokens or 1% of pretraining, and keeps a snapshot of the weights every 25 steps.
The snapshots show how much fine-tuning is actually needed, so I didn't have to guess.

It took about 23 minutes on one H100. The first step's gradient norm was 1.49, the first gradient the model had ever
seen from positions beyond 1024; clipping at 1.0 caught it. Within 40 steps it settled at about 0.28, close to the end
of pretraining (about 0.25). Loss on held-out long pages went from 3.13 to 3.03, most of it in the first 50 steps.

### Results after 200 fine-tuning steps

After 200 steps, the perplexity of a 4096-token page is 21.0, against 23.2 zero-shot: 2.2 lower, with a 95%
interval of [2.1, 2.3]. The short-context cost is gone too. At 1024 tokens the fine-tuned model scores 20.5,
against 22.4 zero-shot and 20.6 for the original model.

The most useful comparison is with Model B as originally trained. The original Model B reads the first
1024 tokens of these pages at a perplexity of 20.6, and can't read further. The fine-tuned model reads all 4096 at
21.0, and its loss at positions 2048–4096 (3.05) is within 0.03 of the original model's at 512–1024. So the model gets four times
the context at nearly the same per-token quality, for 1% of the original training tokens.

Some caution is needed here. The fine-tune trained on exactly this kind of text, the opening 4096 tokens of
long pages, so some of its gain is plain adaptation to the page type: it's better than the original even at the
first 256 tokens (21.1 against 21.4), where position plays no part. Also, the per-position loss stays level rather
than falling past 1024, for the fine-tuned model as for the original one within its own range. Later tokens in these
pages aren't easier to predict, so a level curve is what a model reading the whole page should show, but it can't
prove how much of the distant context the model actually draws on. That needs a sharper test: score the same late
positions with the context cut back to 1024 tokens, and see how much worse they get.

![Perplexity at 1024 and 4096 tokens against fine-tuning steps](/assets/s1/positions/figures/yarn_finetune_steps.png)

**How to read it.** Each point is one snapshot of the fine-tune, scored on the same 500 pages; "zero-shot" is YaRN
before any training. The teal line is the perplexity of the whole 4096-token page; the gold line counts only its
first 1024 tokens, the length the model was originally trained on. Both come from the same pass over the same pages.
The dashed line is Model B before YaRN on those same first 1024 tokens: gold above it means stretching has cost the
model quality inside its old range, gold below it means that cost is gone.

Most of the gain comes early. The first 25 steps (12M tokens, about 0.1% of pretraining) take whole-page
perplexity from 23.2 to 21.5, 77% of the total gain (1.7 of 2.2); by step 100 it's 21.1 (94%), and the last 100
steps add almost nothing. The gold line is back at the original model's quality by step 75. Of the two ways of scaling
the paper's budget down, matching the share of pretraining (about 20 steps) turned out to be the better guide to where the curve bends.

## Caveats, and what to expect at larger scale

The inference numbers come from one model size, two machines and float32 without kernel fusion or CUDA graphs, all of
which push a 124M model towards being overhead-bound. The position analysis covers one context length. Here's what
should carry over to larger models and longer contexts, and what each expectation rests on:

| Finding at 124M | With bigger models | With longer contexts | Basis |
|---|---|---|---|
| KV cache 6.4× smaller | Same ratio per layer; the absolute size grows with $$d·L$$ | The ratio grows (≈19× at 32k for a 7B-class model with Gemma 3's layout) | Arithmetic |
| Decode speed depends on the machine; overhead-bound on H100 | Becomes memory-bound (a 7B model's weights alone take ~4 ms to read per step), so the smaller cache pays | Cache traffic dominates | Arithmetic + both measurements |
| Reading the cache: 1 FLOP per byte for A, 4 for B | Unchanged: it depends only on $$n_h/n_{kv}$$ | Unchanged, but the cache grows | Arithmetic |
| RoPE with base 10,000; 18 of 32 bands turn within the context | — | The base must rise (Llama 3: 500k; Gemma 3 global: 1M), or bands must be rescaled (YaRN) | Arithmetic + papers [[3]](#ref-3), [[9]](#ref-9), [[10]](#ref-10) |
| YaRN at 4×: zero-shot removes the breakdown; 25 fine-tuning steps (~0.1% of pretraining) give most of the gain | The paper's own results are at 7B and above; the 0.1% budget matched here too | Larger stretches need more fine-tuning, or the base raised in pretraining | One model, one factor, one seed |
| Model B's induction heads sit in a local layer, so it can't copy from 768 tokens back | Unknown: larger hybrid models may place copying heads in global layers | The window caps copying unless global layers learn it | One seed per model; a 64-token block |
| A sinks attention on position 0, B (RoPE) on separator tokens | Attention sinks are widespread at scale (Xiao et al., 2023) [[6]](#ref-6) | Sinks on a token need that token within reach of local layers | 32 windows, both models |

A second seed of Model B could, for instance, place its induction head in a global layer, which would change the
copying result. The attention analysis is also descriptive: it shows where attention goes. Showing that the model needs it there takes
a causal check, such as switching a head off and measuring what breaks, which is the natural next step.

## Appendix

### Attention and FlashAttention

This works through the numbers behind the short note on attention's quadratic cost.

All of this has been counted per token, and per token, attention is *linear* in context length: a token at position $$p$$
compares itself with $$p + 1$$ keys, so it costs $$4d·(p + 1)$$ FLOPs. That's why the formula uses the average, $$4d·\bar{k}$$. The
famous quadratic appears when you add up a whole sequence. $$T$$ tokens, each attending to everything before it, make $$T(T + 1)/2$$ query-key pairs, about $$2d·T^2$$ FLOPs per layer. Both are true; one counts per token, the other per sequence.
For the 7B-class model with full attention, the attention FLOPs for one sequence grow from about 0.3 TFLOP at 1k
tokens to about 4,500 TFLOP at 128k: 128 times the context, 16,000 times the work.

FlashAttention (Dao et al., 2022) [[11]](#ref-11) computes exact attention, so the arithmetic stays the same, and its backward pass
even recomputes the scores instead of storing them. What it reduces is memory traffic. Instead of writing the $$T × T$$
score matrix to GPU memory and reading it back twice, it works in tiles that fit on-chip and fuses score, softmax and
weighted sum into one kernel. Memory grows with $$T$$ instead of $$T^2$$, and the time saved is time spent waiting on memory,
the memory-bandwidth case from the three speed limits in the KV cache section.

Our own runs show why that matters even at 1024 tokens. With 12 heads and 60 sequences per GPU, the score matrices
for one layer would take about 1.5 GB in bf16, and keeping all 12 layers' for the backward pass would take about
18 GB. At 128k tokens, one head's scores for one sequence would be 34 GB, more than a third of an H100's memory.
Without FlashAttention, long-context training simply wouldn't fit.

So FlashAttention makes quadratic attention practical, and the cost stays quadratic. Within exact attention, only
restricting which keys a query sees changes the order itself, as Model B's local windows do: a local layer costs $$O(T·W)$$ per sequence, linear in $$T$$. In time, that saving needs a kernel that skips the blocks outside the window, like
FlexAttention's block masks or FlashAttention's window option. With an ordinary dense mask, all $$T^2$$ scores are
computed and most thrown away, which is why the dense-mask version in Part 1's ladder trained *slower*; FlexAttention recovered about 5% of that.

The same holds for decoding with a KV cache: each new token costs $$O(T)$$, one pass over the
cache, so generating a whole sequence still adds up to $$O(T^2)$$, just spread across the steps.

### How the band measures are computed

The measures behind the band-usage heatmap and the three real heads in the position section.

**From the weights.** Write $$W_q^{(j)}$$ for the rows of the head's query projection
that produce band $$j$$'s two dimensions, and $$W_k^{(j)}$$ for the same rows of the key projection it shares under
grouped-query attention, each with the RMSNorm gain in front of attention folded in. The head's **q·k weight** in band $$j$$ is the product of their sizes, as a share of the head's total; its **slow-band weight** sums that share over the 14
slow bands, and a head that treated every band alike would get 14/32, or 44%:

$$w_j = \big\lVert W_q^{(j)} \big\rVert \cdot \big\lVert W_k^{(j)} \big\rVert, \qquad
s_j = \frac{w_j}{\sum_{l=1}^{32} w_l}, \qquad
\text{slow-band weight} = \sum_{j \,\in\, \text{slow}} s_j$$

where the norms are Frobenius norms.

**On real text.** Using the complex form from the RoPE derivation, write $$z_j^q(m) = q_j(m) + \mathrm{i}\,q_{j+h/2}(m)$$ for band $$j$$ of the query at position $$m$$, *before* RoPE's rotation, and $$z_j^k(n)$$ likewise for the key. Band $$j$$'s average contribution
to the score at distance $$\Delta$$, over a window of $$T = 1024$$ tokens, is

$$S_j(\Delta) = \frac{1}{T - \Delta} \sum_{n=0}^{T-1-\Delta}
\frac{1}{\sqrt{h}}\, \mathrm{Re}\Big[\, z_j^q(n+\Delta)\; \overline{z_j^k(n)}\; e^{\,\mathrm{i}\,\Delta\,\theta_j} \Big]$$

averaged over 64 held-out windows. The factor $$e^{\mathrm{i}\Delta\theta_j}$$ is RoPE's rotation at that distance; everything else is
content. The figure's bottom row is the head's mean score, $$S(\Delta) = \sum_j S_j(\Delta)$$. The top row sums $$S_j(\Delta)$$ over each speed
group: fast, middle and slow.

### Code and models

The models are public on Hugging Face: [Model A](https://huggingface.co/CuriosLM/gpt2-124m-fineweb-edu),
[Model B](https://huggingface.co/CuriosLM/gemma-style-113m-fineweb-edu) and the
[YaRN-extended Model B](https://huggingface.co/CuriosLM/gemma-style-113m-fineweb-edu-yarn4k) (4096-token context, after the 200-step
fine-tune). The KV cache, the attention-weights path, the YaRN fine-tune and the analysis behind every figure are in the
[CuriosLM repository](https://github.com/Shinibali/CuriosLM); its README explains how to run them.

### Acknowledgement

Everyone I acknowledged in [Part 1](/gpt2-to-gemma-part-1/#acknowledgement) shaped this part too, especially Percy Liang's Stanford course
[*CS336: Language Modeling from Scratch*](https://cs336.stanford.edu/). Horace He's [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html) gave me
the way to think about generation speed here: compute, memory bandwidth or overhead. Aleksa Gordić's
[*Inside the Transformer: The Life of a Token*](https://www.aleksagordic.com/blog/transformer) is the best place to read about RoPE, YaRN and hybrid attention in depth.

Platforms and tools I used for this part:

- I rented H100s on [RunPod](https://www.runpod.io/) for the inference benchmark and the YaRN fine-tune.
- [Weights & Biases](https://wandb.ai/) tracked the fine-tuning run.
- [Hugging Face Hub](https://huggingface.co/) hosts all three models, including the YaRN-extended Model B.
- [Claude Code](https://claude.com/claude-code), Anthropic's coding assistant, worked alongside me: it helped build and test the KV cache and the analysis scripts, debug the fine-tune setup, and draft and fact-check this article.

If you spot a mistake, or have comments or critiques, I'd be glad to hear them on
[LinkedIn](https://www.linkedin.com/in/shinibali-bhattacharyya/) or [X](https://x.com/ShinibaliB).

### References

1. <span id="ref-1"></span>Su et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
2. <span id="ref-2"></span>Barbero et al. (2025). [Round and Round We Go! What Makes Rotary Positional Encodings Useful?](https://arxiv.org/abs/2410.06205).
3. <span id="ref-3"></span>Peng et al. (2023). [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071).
4. <span id="ref-4"></span>Muennighoff et al. (2023). [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264).
5. <span id="ref-5"></span>Olsson et al. (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
6. <span id="ref-6"></span>Xiao et al. (2023). [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453).
7. <span id="ref-7"></span>Dosovitskiy et al. (2020). [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).
8. <span id="ref-8"></span>Holtzman et al. (2019). [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751).
9. <span id="ref-9"></span>Gemma Team (2025). [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786).
10. <span id="ref-10"></span>Llama Team (2024). [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).
11. <span id="ref-11"></span>Dao et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).
12. <span id="ref-12"></span>Horace He (2022). [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html).
13. <span id="ref-13"></span>Jain & Wallace (2019). [Attention is not Explanation](https://arxiv.org/abs/1902.10186).
14. <span id="ref-14"></span>Elhage et al. (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
15. <span id="ref-15"></span>Clark et al. (2019). [What Does BERT Look At? An Analysis of BERT's Attention](https://arxiv.org/abs/1906.04341).
16. <span id="ref-16"></span>Wang & Chen (2020). [What Do Position Embeddings Learn? An Empirical Study of Pre-Trained Language Model Positional Encoding](https://arxiv.org/abs/2010.04903).

### Citation

```bibtex
@misc{bhattacharyya2026gpt_to_gemma_2,
  author       = {Bhattacharyya, Shinibali},
  title        = {GPT-2 (124M) to Gemma 3-Style (113M) language model, Part 2: KV Caches, Attention Visualization, RoPE Clocks and 4× Context Extension},
  year         = {2026},
  url          = {https://shinibali.github.io/gpt2-to-gemma-part-2/},
}
```
