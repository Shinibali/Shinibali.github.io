---
layout: post
title: "GPT-2 (124M) → Gemma 3-Style (113M), Part 2: KV Caches, Attention Visualization, RoPE Clocks and 4× Context Extension"
date: 2026-10-14
math: true
---

---

## 0. TL;DR

Part 1 trained a GPT-2 and a Gemma 3-style model of the same size on the same 9.8 billion tokens and found them tied on
quality, with the Gemma recipe slightly slower to train. This part looks at what the recipe changes once the models are
trained:

- **A 6.4× smaller KV cache doesn't automatically mean faster generation.** On a laptop GPU the Gemma 3-style model
  generates up to 1.55× faster. On an H100 it's a third slower, because a model this small is limited by per-step overhead,
  and it has more operations per step. The smaller cache pays off at serving scale: big batches, long contexts, bigger
  models.
- **The two models encode position very differently and end up in similar places.** RoPE gives Model B 32 fixed
  "clocks"; Model A's learned position table organizes itself into slow waves. Both end up encoding position as distance.
- **The windows cost one specific skill.** Both models copy from earlier in the text using induction heads, but
  Model B's live in a local layer: with the repeat 768 tokens back, Model A still copies and Model B can't. On
  ordinary text the two models disagree little, and B's position scheme even changes where it parks unneeded
  attention.
- **YaRN stretches Model B from 1,024 to 4,096 tokens without retraining.** Zero-shot, it removes the breakdown
  (perplexity 37.7 → 23.2 at 4,096). Continuing the training run for 200 steps, 1% of pretraining, brings it to 21.0
  at 4,096 tokens, close to the original model's 20.6 at 1,024; 25 steps already deliver three quarters of that.

## 1. Where Part 1 left off

[Part 1](/gpt2-to-gemma-part-1/) compared two 124M-class models trained identically on FineWeb-Edu:

- **Model A** is GPT-2 as nanoGPT ships it: learned position embeddings, full multi-head attention.
- **Model B** is Gemma 3-style [[9]](#ref-9): RoPE [[1]](#ref-1), grouped-query attention (12 query heads sharing 3 key/value heads), a GeGLU MLP,
  sandwich RMSNorm, and local 512-token attention in three layers out of four.

They tied on loss and on every benchmark. On paper, Model B is cheaper in FLOPs; in practice, it trained 4% slower.

What Part 1 couldn't answer is what the Gemma recipe is *for*. Its two attention changes, GQA and local windows, were
designed to shrink the memory a model needs while it generates text, and RoPE was designed to make position relative,
and extensible. This part measures all three: the cost of generating text, how each model represents position, and
how far Model B's context can be stretched.

## 2. Inference: where the smaller cache pays, and where it doesn't

Part 1's ladder pointed at the recipe's intended payoff, a much smaller KV cache. Measuring it needed a KV cache, so
I built one. The result depended on the machine more than on the model.

### A KV cache, briefly

A language model generates one token at a time, and each new token attends to every token before it. Without a cache,
every step reprocesses the whole sequence from scratch, which is wasteful because the keys and values of earlier
tokens never change. A KV cache stores them once and reuses them, so each step only has to process the one new token.

The idea is simple. Getting it right for every rung of the ladder took more care: positions, rotated keys,
grouped key/value heads, window eviction and the causal mask each need handling (details in the appendix,
[What the KV cache has to get right](#what-the-kv-cache-has-to-get-right)). Only the 3 key/value heads are
cached, which is the 4× saving, and a local layer's cache stops growing at 511 positions.

The definition of correct is simple: at every step, the cached model must produce the same logits as recomputing the
whole sequence. The tests check exactly that, for all five rungs, for single tokens, for chunks, and past the window
limit.

### Three things that can limit speed

Horace He's [*Making Deep Learning Go Brrrr*](https://horace.io/brrr_intro.html) [[12]](#ref-12) sorts GPU time into three buckets:

- **Compute:** the chip is busy doing arithmetic.
- **Memory bandwidth:** it's waiting for data to arrive from memory.
- **Overhead:** it's waiting for the next instruction. That covers launching kernels, Python, and moving small
  tensors around.

Which bucket dominates decides whether a smaller KV cache makes a model faster.

**Prefill** reads the prompt: every prompt token goes through the model in one forward pass, and their keys and values
are *written* to the cache. With enough tokens that's large matrix multiplications, so prefill leans towards
*compute*. Model B does 9% less arithmetic per token than Model A (Part 1, "Counting FLOPs"), so its smaller cache doesn't help
here.

**Decode** generates the answer one token per step. Each step does very little arithmetic, but it has to read every
weight and every sequence's whole cache. For a large model that makes decode *memory-bound*, which is exactly where a
smaller cache pays off. For a small model there's a catch: each step also pays a fixed cost to launch a few hundred
small kernels, one per operation per layer. If that fixed cost is bigger than the time spent moving data, decode is
*overhead-bound*, and the cache size barely matters.

If you've used an LLM API with "cache write" and "cache read" prices, it's the same split: prefill writes the cache,
decode reads it.

### The numbers, on two machines

The same benchmark ran on a MacBook (M1 Pro) and on one H100, both in float32, with warm-up runs, synchronization
before reading the clock, and the median of repeats. Each generation run produces 256 new tokens.

| | Model A | Model B | B vs A |
|---|---|---|---|
| KV cache per sequence, 1,024 tokens (any machine) | 75.5 MB | 11.8 MB | 6.4× smaller |
| **Mac:** generation with cache, batch 8, 512-token prompt | 297 tok/s | 461 tok/s | **1.55×** |
| **Mac:** prefill, batch 8, 1,024 tokens | 13.5k tok/s | 12.2k tok/s | 0.90× |
| **H100:** generation with cache, batch 8, 512-token prompt | 1,988 tok/s | 1,290 tok/s | **0.65×** |
| **H100:** prefill, batch 8, 1,024 tokens | 212k tok/s | 194k tok/s | 0.92× |

At 1,024 tokens, prefill behaves the same on both machines: B is 8–10% slower, in line with Part 1's training slowdown
(on the H100, short single-sequence prompts are slower still, because they are overhead-bound).
Generation flips. On the Mac, B generates up to 1.55× faster (batch 8, 512-token prompt); on the H100, it's a third *slower*.

The time per decode step shows why:

![Decode step time against megabytes read per step, on each machine](/assets/s1/efficiency/figures/decode_regimes.png)

**On the Mac, step time rises with the work done per step.** Each step reads all the weights plus the whole cache,
between 457 and 875 MB across the eight settings, and the bytes read alone explain 84% of the variation in step time.
More sequences and longer prompts mean more cache to read, and B reads less of it, so B wins in all but the smallest setting, and by more as
the batch grows.

**On the H100, step time is flat.** Model A takes 3.8–4.0 ms per step and Model B 6.1–6.3 ms, whether it's generating
for one sequence or eight, from a 128-token prompt or a 512-token one. Reading 460–880 MB at the H100's 3.35 TB/s
takes 0.14–0.26 ms, so memory traffic is a rounding error here. What's left is overhead: a few
hundred kernel launches per step, each costing microseconds, adding up to milliseconds. Model B launches more of them,
with two extra normalizations per block, the gating multiply, the rotations and the local-window masks. So on the
H100 it's slower, and its smaller cache never gets a chance to matter.

![Generation speed with and without the cache, and prefill speed, on the Mac](/assets/s1/efficiency/figures/inference_speed.png)

The cache itself follows the same logic. On the Mac it speeds up generation 2.5–5.3× at batch 1 and 8–23× at batch 8,
because recomputing the whole sequence every step is a lot of extra work. On the H100, at batch 1 with a short prompt,
the cache gives no speed-up at all (0.9–1.0×), because recomputing 128 tokens costs no more launch overhead than
processing one. Its benefit only appears with longer prompts and bigger batches (up to 6.3× for Model A at batch 8).

![KV-cache memory as the context grows](/assets/s1/efficiency/figures/kv_cache.png)

The memory plot shows the two savings separately. GQA divides the slope by four. The local windows make most of the
layers stop growing at 511 positions, so past that point only B's three global layers keep adding to the cache. The
measured sizes match the formula exactly, on both machines.

### What this means

The arithmetic says when the H100 result changes. For cache traffic to outweigh about 4 ms of overhead, a step has to
read around 13 GB, roughly 175 full-length sequences of Model A's float32 cache at once. Real serving systems work at
that scale: large batches, long contexts, bigger models, and CUDA graphs and kernel fusion that cut the per-step
overhead. A 124M model on a single H100 isn't there, which is worth knowing before reading any small-scale inference
benchmark, including this one.

### At serving scale

Arithmetic intensity, the FLOPs done per byte read, says which part of decoding can ever become compute-bound. For the
weights, each parameter read in bf16 (2 bytes) is used for 2 FLOPs per sequence in the batch, so the intensity is
about the batch size. An H100 does about 295 bf16 FLOPs per byte it can read, so decode only becomes compute-bound
for the weights at batches of a few hundred sequences. Attention over the cache is different. Each cached key and
value is read for one sequence only, and each one serves n_h / n_kv query heads. That gives an intensity of 1 FLOP
per byte for Model A and 4 for Model B, *whatever the batch size*. Reading the cache is memory-bound at any scale,
and GQA's real job is to multiply that intensity by the number of query heads per key/value head.

The same formula gives the KV cache per sequence of a 7B-class model (32 layers, d_model 4,096) in
bf16:

| Context | Full multi-head attention | GQA (8 KV heads) | GQA + Gemma 3-style local windows |
|---|---|---|---|
| 8k tokens | 4.3 GB | 1.1 GB | 0.31 GB |
| 32k tokens | 17.2 GB | 4.3 GB | 0.91 GB |
| 128k tokens | 68.7 GB | 17.2 GB | 3.3 GB |

At 32k tokens, one sequence's full-attention cache would fill a fifth of an 80 GB H100. The Gemma 3-style layout fits
nearly twenty sequences in the same space. Overhead stops dominating too: reading a 7B model's bf16 weights alone
takes about 4 ms per decode step at 3.35 TB/s, as long as the whole fixed overhead measured here. That's the regime
the Gemma design targets, where a 6.4× smaller cache means 6.4× more sequences in the same memory, and where the
Mac's direction of travel, not the H100's 124M result, is the one that carries over.

## 3. What changes as the context grows

Part 1 counted each model's FLOPs per token, term by term. One of those terms, attention, is the only one that grows
with context length, so the algebra also says what long contexts do to a model.

Per layer, attention costs 2d(T + 1) FLOPs per token against the MLP's 16d², so **attention overtakes the MLP when
T ≈ 8d**, with d the model dimension (d_model). For our d_model (768) that's about 6,000 tokens; for a 7B-class
model (d_model 4,096), about 33,000. A local layer's queries see at most W keys, so its attention cost levels off at
4d·W per token. That only reaches the MLP's 16d² if W > 4d (about 3,000 tokens at our d_model), so with Model B's
512-token window the local layers never cross over; only the global layers do.

To see what that means in practice, take the two recipes from Part 1, scale each to a 7B-class and a 14B-class
model, and count the forward FLOPs per token at a 1k and a 128k context. These are absolute FLOPs, not shares:

![Forward GFLOPs per token for Model A's and Model B's recipes, at two sizes and two context lengths](/assets/s1/efficiency/figures/flops_by_context.png)

| Forward GFLOPs per token | 1k context | 128k context | Attention scores' share at 128k |
|---|---|---|---|
| Model A's recipe, 6.7B | 13.6 | 47.7 (3.5×) | 72% |
| Model A's recipe, 15.4B | 31.2 | 95.1 (3.0×) | 68% |
| Model B's recipe, 5.8B | 11.9 | 20.5 (1.7×) | 43% |
| Model B's recipe, 13.5B | 27.3 | 43.4 (1.6×) | 38% |

With full attention, stretching the context from 1k to 128k tokens multiplies the work per token by 3–3.5×, almost
all of it attention scores. With Model B's recipe (512-token windows in three layers of four), the same stretch
costs 1.6–1.7×, because only the global layers pay for the long context. The result: at 128k tokens, the 13.5B model
built with Model B's recipe does *less* work per token than the 6.7B model built with GPT-2's (43 against 48 GFLOPs).
That's on top of the memory savings in Section 2. Gemma 3's report [[9]](#ref-9) gives memory as its main reason for going to five
local layers in six at 128k, but the arithmetic points the same way.

The same counting matters for how training efficiency is reported. The PaLM formula behind Part 1's logged MFU
charges every token for attending to the whole context; a causal layer computes about half of that, and a local
layer only its window. At 1,024 tokens, for the 7B-class models above, the formula overstates the work by 2–3%. At
128k tokens it overstates it by 72% for Model A's recipe and by a factor of 3.9 for Model B's, because it charges
the local layers for the full 128k they never look at. A windowed long-context model logging "40% MFU" this way is
really using about 10% of its GPUs. MFU numbers for long-context or hybrid-attention runs only mean something
alongside the FLOP count they assume.

### Isn't attention quadratic? And doesn't FlashAttention fix that?

Per token, attention is linear in context length; summed over a whole sequence it's the familiar quadratic,
about 2d·T² FLOPs per layer. FlashAttention (Dao et al., 2022) [[11]](#ref-11) doesn't change that arithmetic. It cuts memory
traffic, so the T × T score matrix never has to sit in GPU memory, which is what makes long-context training fit
at all. Within exact attention, only restricting which keys a query sees, as Model B's local windows do, changes the order, to O(T·W)
per sequence, and only with a kernel that skips the masked blocks. The appendix
([Attention, FlashAttention and the quadratic](#attention-flashattention-and-the-quadratic)) works through
the numbers.

## 4. Inside the models: how each one handles position

The two models end with the same loss, but they get there with very different machinery for the one thing a
transformer can't know by itself: where each token sits in the sequence. Model A learns a vector for every position
from scratch. Model B is told position through a fixed rotation. It's worth looking at what each actually does.

### RoPE is a set of clocks

RoPE [[1]](#ref-1) splits each 64-dimensional attention head into 32 pairs of dimensions and rotates each pair by an angle
proportional to the token's position. Each pair turns at its own speed. The fastest completes a full turn every 6
positions. The slowest would need 47,000 positions to complete one. When a query and a key are compared, only the
*difference* of their angles survives, so attention depends on how far apart two tokens are, not on where they are.
In effect, each head reads the distance between two tokens off a set of 32 clocks running at different speeds.

![Wavelength of each RoPE band against the 1,024-token training context](/assets/s1/positions/figures/rope_bands.png)

Only 18 of the 32 clocks complete a full turn within the 1,024-token training context. The other 14 never do: in
training, the model only ever sees a fraction of their cycle. That's harmless at 1,024 tokens, and it's precisely the
problem when you try to run the model on longer inputs. It's what YaRN fixes in Section 6.

### How a head uses the clocks

Knowing that RoPE rotates pairs is one thing. Seeing what that does to attention takes one more step: treat each
pair as a complex number.

Take one head, with head size h = 64, and band i: dimensions i and i + h/2, which RoPE rotates by m·θ_i at position
m, where θ_i is the band's fixed speed. The same happens to the key at position n.

**Step 1: a rotation is a multiplication.** Write the query's pair as one complex number,
z_i^q(m) = q_i(m) + i·q_{i+h/2}(m), and the key's as z_i^k(n). Rotating a 2-D vector by an angle α is the same as
multiplying its complex number by e^{iα}. So RoPE turns the query's pair into z_i^q(m)·e^{i m θ_i}, and the key's into
z_i^k(n)·e^{i n θ_i}.

**Step 2: a dot product is the real part of a product.** For two 2-D vectors written as complex numbers a and b,
their dot product is Re[a·b̄], where b̄ is the complex conjugate. So band i's share of the attention score is

$$\frac{1}{\sqrt{h}}\,\mathrm{Re}\Big[\, z_i^q(m)\,e^{i m \theta_i} \cdot \overline{z_i^k(n)\,e^{i n \theta_i}} \,\Big]
= \frac{1}{\sqrt{h}}\,\mathrm{Re}\Big[\, z_i^q(m)\,\overline{z_i^k(n)} \; e^{\,i (m-n)\,\theta_i} \Big]$$

because the conjugate of e^{inθ_i} is e^{−inθ_i}, and the two rotations combine into one. **Only the difference
Δ = m − n survives.** This is the line that makes RoPE relative.

**Step 3: separate content from position.** The product c_i = z_i^q(m)·conj(z_i^k(n)) depends only on what the
query and key vectors contain, not on where they are. Write it as a size and an angle, c_i = |c_i|·e^{iφ_i}. Summing
over the 32 bands gives the whole score:

$$\text{score}(\Delta) = \frac{1}{\sqrt{h}} \sum_{i=1}^{32} |c_i| \cos\!\big(\Delta\,\theta_i + \varphi_i\big)$$

The result is worth reading slowly. The attention score, as a function of the distance Δ between two tokens, is a sum
of 32 waves. **The waves' frequencies are fixed by RoPE. How much of each wave a head uses, and where each wave
peaks, are set by content**, through the query and key weights the head has learned. A head doesn't get told how to
treat distance; it builds its own preference by mixing fixed waves, like an equaliser setting the volume of fixed
frequency bands.

In practice, that gives a head three kinds of behaviour to choose from. Fast waves give a sharp preference for nearby
tokens, just as a thin line needs a fine brush; middle waves a smooth fade over tens to hundreds of tokens; and slow
waves, the ones that never complete a turn within the window, almost none, so a head built on them decides on *what*
the tokens are, not where they are.

Do Model B's trained heads actually split like this? To check, it helps to name the three speeds precisely. The
cleanest way is the one the YaRN paper [[3]](#ref-3) uses: count how many full turns each band makes within the 1,024-token
training context.

- **Fast bands** make 32 turns or more (6 of the 32 bands).
- **Middle bands** make between 1 and 32 turns (12 bands).
- **Slow bands** never complete a single turn (the other 14).

These are the same groups YaRN treats differently in Section 6: it leaves the fast bands alone, rescales the slow
ones, and blends the middle. The thresholds are counted in turns, so they mean the same thing at any context length,
but which bands they select shifts with the context. (The 1-turn line is principled; the 32-turn line is the YaRN
authors' empirical choice for Llama models.)

The first figure reads the answer from the trained weights alone. A head can only use a band as much as its query and
key projections feed that band's pair of dimensions. The head's **q·k weight** in band i is the size of the query and
key projection rows that feed band i, multiplied together, as a share of the head's total. Its **slow-band weight** sums
that share over the 14 slow bands; a head that treated every band alike would get 14/32, or 44%. (The formula is in the
appendix, [How the band measures are computed](#how-the-band-measures-are-computed).)

![Which RoPE bands each of Model B's 144 heads uses, measured from the trained weights](/assets/s1/positions/figures/band_usage.png)

Each row is one head (12 per layer, 12 layers) and each column one band, fastest on the left. Darker means more of
that head's q·k weight. The dashed line separates the bands that turn within the context from the slow ones. The
pattern leans right:

- **Overall:** the median slow-band weight is 53%, against 44% for a head that treated every band alike.
- **Five heads have a slow-band weight above 75%:** one in layer 1, three in layer 6 and one in layer 8. They're the
  dark stripes on the right.
- **No head concentrates on the fast bands.** The lowest slow-band weights, mostly in layer 0, sit at or just below the 44%
  of an even spread (lowest 42%).

This matches what Barbero et al. (2025) [[2]](#ref-2) found in a much larger Gemma model: a strong preference for the lowest
frequencies, which they read as carrying meaning, and the highest frequencies used for positional patterns.

Weights show what a head *can* use. The second figure shows what it *does* use, on real text. It runs held-out
FineWeb-Edu through Model B and, for three heads, averages the pre-softmax score q·k/√h over every query–key pair the
same distance apart, so higher means the head prefers keys at that distance.

Using the complex form from the derivation above, that average score splits exactly into one contribution per
band (formula in the appendix). The figure's bottom row is the head's mean score at each distance, and the top
row sums the bands' contributions within each speed group: fast, middle and slow.

![Three real Model B heads: average attention score vs distance, and the contribution of fast, middle and slow bands](/assets/s1/positions/figures/real_heads.png)

**How to read it.** Each column is one real head, chosen from the heatmap:

- the highest slow-band weight (layer 6, head 11, 84%);
- the median (layer 9, head 1, 53%);
- the lowest (layer 0, head 6, 42%).

Panel titles mark global or local layers; local heads' curves stop at W = 512. The bottom row is the head's average
score at each distance, and the top row splits it into the parts from its fast, middle and slow bands. The three parts
add up exactly to the bottom row: I checked the split against the model's own attention scores, and they agree to
every printed digit.

The three heads match the three behaviours:

- **Layer 0, head 6 is sharply local.** Its score halves from distance 0 to distance 1 and keeps falling. That first
  drop is spread across all three speed groups.
- **Layer 9, head 1 fades smoothly** over about 100 tokens, through its middle and slow bands together.
- **Layer 6, head 11 barely cares about distance** beyond its immediate neighbours: its score is nearly flat from 3
  tokens out to the edge of its 512-token window, and its fast bands contribute nothing at all.

One caution in reading these curves. They mix two things: RoPE's rotations, and the fact that nearby tokens tend to be
similar in content. The steep drop over the first few tokens in the slow bands, in layer 6's head and layer 0's, can't
come from rotation, since those bands barely turn in a few tokens. It's content: a query matching keys from its own
neighbourhood. The flat stretch beyond that is where the slow bands' indifference to position shows.

### Learned positions turn into waves anyway

Model A's position table is just 1,024 free vectors. Nothing in its design says anything about distance, smoothness
or waves. Training discovers structure anyway. Model B has no such table, since RoPE rotates queries and keys instead
of adding a vector to each token, but one thing can be compared directly: how similar two positions look as a
function of their distance.

![Position as waves: Model A's learned table, compared with Model B's RoPE](/assets/s1/positions/figures/learned_positions.png)

**Left: Model A's position vectors, every one compared with every other.** The bright diagonal band says nearby
positions have similar vectors, and the similarity fades with distance. Distance alone explains 71% of the variation
in this matrix. The model has learned something close to "position as distance", the very property RoPE builds in.

**Middle: the three directions along which Model A's position vectors vary most**, plotted against position.
They're smooth waves: half a cycle, a full cycle, about one and a half. That's no coincidence. A similarity that
depends only on distance has close-to-sinusoidal eigenvectors, the same mathematics as a Fourier transform.

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

The same split shows up if Model A's table is broken into waves along the position axis: two-thirds of its
variation sits in a handful of slow waves, 341 positions and longer, and most of the rest, about 22%, is a flat,
noise-like floor at short wavelengths. RoPE puts structured waves exactly there, from 6 positions up, and keeps 14
bands slower than the whole window, which a table of 1,024 rows can't represent at all.

That's one plausible reason RoPE was the one change in Part 1's ladder that improved loss, and it's why YaRN, in
Section 6, has slow bands to rescale.

### At longer contexts

The two schemes part ways at longer contexts. Model A's position table simply stops at 1,024 rows, so it
can't read further at all. Building one for a 128k context in a 7B-class model would take 537M parameters, each row
trained only on the tokens that happen to land at its position. That goes a long way toward explaining why learned position
tables have all but disappeared from long-context models.

RoPE needs adjusting too. With the base of 10,000 used here, every one of a 128-dimensional head's 64 bands completes
a turn within 128k tokens, so no band is slow enough to tell "far" from "very far". Long-context models raise the
base: Llama 3 [[10]](#ref-10) uses 500,000, and Gemma 3 uses 1,000,000 on its global layers. That leaves 15 and 17 bands slower than
the whole 128k context. Model B's windows save more and more memory as the context grows, but more and more of the
context also lies outside them. In Gemma 3's layout, five layers in six never look beyond 1,024 tokens, even at a
128k context, so everything long-range has to travel through the global layers. That's the price of the memory savings, and why
the design keeps some global layers at all.

## 5. Seeing attention

This section looks at what each model attends to while it reads. Two cautions first. Attention weights show where
information *can* flow, not why a model makes the prediction it does. And a weight averaged over a layer's heads can
hide a single head doing something sharp. The views below are a way to look, not a proof.

**Getting the weights.** The fused kernels (PyTorch's scaled-dot-product attention, and FlexAttention for Model B's
windows) never form attention probabilities as a tensor. So the model gets an evaluation-only path that computes
softmax(q·kᵀ/√h + mask) step by step, with the same masks and key/value grouping; a test checks that it reproduces
the fused output exactly.

A small example shows why attention is worth looking at in a byte-level model. GPT-2's tokenizer works on bytes, and
an emoji is four bytes of UTF-8. In "Learning is Fun 🤗, Keep exploring 🔭", each emoji becomes three tokens: one they
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
property of greedy decoding (Holtzman et al., 2019) [[8]](#ref-8) rather than of either model. Sampled instead, with the same
settings the sampling script uses, both write ordinary on-topic text: Model A continues "for more learning! 🔨 This is
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

The copy stripe is strongest in one layer per model (with a weaker echo in layer 8): layer 7 in Model A, layer 6 in Model B, the layers that hold each
model's strongest induction head below. The other layers do simpler things. Model B's layer 0 mostly attends to the
token itself and its layer 2 to the previous token, the kind of local bookkeeping a copying head needs as an input.
Both models also park a lot of attention on the opening `<|endoftext|>` token, Model A especially in its later
layers.

One sentence can't say much about two 12-layer models, and it never reaches Model B's windows, so the rest of this
section measures every head on full-length text and tests one mechanism directly.

### How far each layer looks

![Mean attention distance and first-token attention for every head of both models](/assets/s1/attention/figures/attention_distance.png)

**How to read it.** Each dot is one head, the line its layer's average; blue is Model A, orange Model B, the shaded
columns Model B's three global layers, and the dashed line Model B's window. The left panel is *attention distance* [[7]](#ref-7):
how many tokens back a head looks on average, weighting each earlier token by its attention (the vision-transformer
measure). It uses only queries in the second half of 32 held-out 1,024-token windows, so every head could look at
least 512 tokens back. The other two panels measure two kinds of *attention sink* [[6]](#ref-6), a fixed place models park
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
rest on position 0 (its local layers can't reach it from these queries), but B does rest on separators, less (at most 11% of a layer, in layers 6, 8 and 10). The
reason is how each model knows where it is. A's learned position table gives position 0 its own vector, so the first
position is recognisable whatever token sits there. RoPE gives position 0 no vector of its own, so B has nothing built in that marks it, and its resting place is a
token it can recognise instead.

### Induction heads, inside and beyond the window

The best-documented circuit in language models is the *induction head* [[5]](#ref-5): at a token it has seen before, it attends to
the token that followed the earlier occurrence, so the model can copy what came next. The standard test removes every
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
(score 0.93), with 5 heads above 0.3; Model B's is layer 6, head 11 (0.90), with 4. Both copy, but not equally well.
The second copy of the block costs Model A 1.6 nats per token and Model B 6.1, against about 13.3 for the unseen
first copy.

With the repeat 768 tokens back, Model A barely notices: the same head still scores 0.90, and the second copy costs
3.0 nats. Model B's induction heads are gone. No head scores above 0.3, its best manages 0.05, and the second copy
costs 12.3 nats, almost as much as tokens it has never seen. Its strongest induction head lives in layer 6, a local
layer, and the earlier copy is now out of that layer's reach. None of its three global layers took over the job.

That's a cost of the local-global recipe that average loss doesn't show. Copying something from far back is a narrow
skill, but a real one, and here the memory savings from the windows cost exactly that.

### Where the two models disagree

Averages can hide where two models part ways, so this compares their predictions token by token on held-out text,
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

The tail is where the models part ways, and it tilts slightly toward B: among the 1,000 strongest disagreements, B
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
48% in B), a fifth to a quarter to "da" and "isy", the word being completed, and the rest across the passage. The difference isn't where the models look but what they make
of it, which attention weights alone can't show.

## 6. Stretching the context: YaRN, 1,024 → 4,096

Model A can never read past 1,024 tokens. Its position table simply has no row 1,025. Model B has no table, so in
principle RoPE can rotate a query to any position. In practice, a RoPE model run past its training length falls apart
quickly. Section 4 shows why: the slow bands reach angles never seen in training, and a head's mix of waves was only
tuned on distances up to 1,024.

[YaRN](https://arxiv.org/abs/2309.00071) is a recipe for fixing this after training. It leaves the fast bands alone,
since they already cycled many times and their behaviour at new positions is familiar. It stretches the slow bands so
that 4,096 positions map into the range of angles the model saw at 1,024, blends smoothly between the two in the
middle, and adds a small correction to the attention temperature. The paper shows it working with no training at
all, and better with a short fine-tune on long sequences.

In Section 4's terms, YaRN's default thresholds leave Model B's 6 fast bands alone, fully interpolate the 14 slow
ones and blend the 12 in between, so more than half of every head's positional machinery is rescaled. Stretching the
context 2×, 4× or 8× without it would make 3, 5 or 7 more bands complete their first full turn, at angles the model
never saw in training.

![Each RoPE band's wavelength in Model B, as trained and with YaRN at 4×](/assets/s1/positions/figures/yarn_bands.png)

**How to read it.** Each dot is one of Model B's 32 RoPE bands, fastest on the left. Hollow dots are the wavelengths
the model was trained with, filled dots the wavelengths YaRN gives it, and the vertical line joining them is how far
YaRN moved that band. The two dashed lines mark the training length (1,024) and the new target (4,096). Purple bands
don't move. Gold bands are stretched the full 4×, and every one of them lands above the 4,096 line. Teal bands are
blended, moving less the faster they are.

The figure shows the point of the recipe in one number. Run plainly at 4,096 tokens, 23 bands would complete a full
turn, 5 more than in training. With YaRN it's 18 again: the same bands that turned within 1,024 tokens during
training are the only ones that turn within 4,096 now. Every band sees a range of angles it already knows.

**Why 4× and not more.** The YaRN paper's [[3]](#ref-3) zero-shot results hold up well to about twice the training length and
degrade beyond it; the long-context results it's known for come after a short fine-tune. At 2×, zero-shot YaRN should
already be close, leaving the fine-tune little to show; at 8×, the target is beyond zero-shot range and pages that
long are rare (about 1% of FineWeb-Edu validation pages reach 8k tokens, against about 3% for 4k). At 4×, 5 bands
newly complete a turn and there's room for the fine-tune to close the gap. Both versions are measured at 4,096
tokens.

### Two choices the paper leaves open

**Which layers to stretch.** Three of Model B's four layers use a 512-token local window, so they never look more
than 511 tokens back at any context length; only the global layers ever meet a distance they weren't trained on.
Production models split on what to do about that. Gemma 3 stretches only its global layers (and with linear
interpolation rather than YaRN). [Rnj-1.5](https://huggingface.co/EssentialAI/rnj-1.5-instruct) and gpt-oss, both
mixing local and global layers, apply YaRN to every layer alike. With no settled convention, Model B follows the paper
and the latter two: one stretched table for every layer. YaRN leaves the fast bands that do most of a local layer's
work untouched anyway.

**The attention temperature.** YaRN also multiplies every query and key by 0.1·ln(s) + 1, which is 1.139 at s = 4,
so attention scores grow by 1.139² ≈ 1.30 and attention gets slightly sharper. It compensates for two things that
flatten attention at the new length: the squeezed slow bands, and four times as many keys to spread attention over.
Rnj-1.5's released config turns this correction off, presumably because a long mid-training stage let the model find
its own sharpness. Model B keeps it: zero-shot there is no training to absorb the change, and keeping the same setting
in both versions means the fine-tune is the only difference between them. Folding the factor into RoPE's cosine and
sine tables applies it to both q and k for free, which is why the paper writes it as a square root.

### Zero-shot: stretching without training

The test set is 500 held-out FineWeb-Edu pages, each at least 4,096 tokens long, each read from its start in one pass.
Every window is therefore one real page, and position p always has exactly p earlier tokens of the same page to draw
on. With causal attention, one pass at 4,096 tokens also gives the answer for every shorter length, since a token's
loss never depends on what comes after it.

![Perplexity against length, and loss at each position, for Model A, Model B as trained, and Model B with YaRN with and without fine-tuning](/assets/s1/positions/figures/yarn_context.png)

**How to read it.** The left panel is the perplexity of a page's first L tokens, the way the YaRN paper reports it;
the right panel is the average loss at each position. The dashed line is the training length. Model A's line stops
there because it has no position past 1,024 to look up. The teal line is the fine-tuned model, which the next part
of this section introduces.

Run past its training length, plain RoPE breaks down. Its loss at each position jumps from about 3.0 to 4.2 between
1,024 and 4,096, and the perplexity of the whole 4,096-token page climbs to 37.7. With YaRN and no training at all, it
stays at 23.2: 14.6 lower, with a 95% interval of [14.1, 15.1] from a paired bootstrap over the pages. The breakdown
is gone.

It isn't free. At 1,024 tokens, where the original model needed no help, YaRN costs perplexity 20.6 → 22.4: the
sharper attention and squeezed slow bands apply at every length, not only the new ones. More telling is what the right
panel shows past the training length: zero-shot YaRN's loss holds level at about where it was before 1,024 (3.15
at positions 2,048–4,096, against 3.13 at 512–1,024), but at that slightly worse level throughout. The model is
stable at 4,096 tokens without being any better for the extra length.

### Fine-tuning: continuing the training run

The paper fine-tuned Llama 2 7B for 400 steps of 64 × 64k-token sequences from long books (about 1.7B tokens,
roughly 0.1% of its pretraining). Carried over to a 124M model trained on 9.8B tokens, those numbers disagree.
Matching the *share* of pretraining gives about 10M tokens, about 20 of Model B's steps. Matching the *steps and
batch* gives about 105M. There is no scaling law for how much a YaRN fine-tune needs, and the paper's learning rate
(2e-5) says little about a 124M model's.

So the fine-tune doesn't borrow the paper's settings. It continues Model B's own training run, changing as little as
possible:

- It starts from the final checkpoint at step 20,000 with the optimizer's state restored, so AdamW's running averages
  carry over.
- The learning rate stays where the cosine schedule ended, at 6e-5, with no new warmup. A run resumed mid-schedule
  doesn't re-warm, and the restored optimizer state already matches the model.
- Each step is still 491,520 tokens, now as 15 sequences of 4,096 instead of 60 of 1,024. Weight decay, gradient
  clipping, precision and seed are unchanged.

Three things do change: the length, YaRN switched on, and the data. The pretraining stream packs pages back to back,
so most 4,096-token windows would join several short pages and seldom reward looking far back. The fine-tune reads
only pages at least 4,096 tokens long, one page per sequence, like the paper's long books: 3% of pages but about a
quarter of the tokens. The model has seen these pages once already: pretraining made
one pass over almost all of the 10B-token sample. Repeating about 1% of the data once is well within what
[Muennighoff et al. (2023)](https://arxiv.org/abs/2305.16264) [[4]](#ref-4) found to be as good as fresh data. The evaluation pages
come from the held-out split, which the model never trained on.

The run is 200 steps, about 98M tokens or 1% of pretraining, and keeps a snapshot of the weights every 25 steps.
Rather than guessing how much fine-tuning is enough, the snapshots measure it.

It took about 23 minutes on one H100. The first step's gradient norm was 1.49, the first gradient the model had ever
seen from positions beyond 1,024; clipping at 1.0 caught it. Within 40 steps it settled at about 0.28, close to the end
of pretraining (about 0.25). Loss on held-out long pages went from 3.13 to 3.03, most of it in the first 50 steps.

### Fine-tuned: what 200 steps buy

After 200 steps, the perplexity of a 4,096-token page is 21.0, against 23.2 zero-shot: 2.2 lower, with a 95%
interval of [2.1, 2.3]. The short-context cost is gone too. At 1,024 tokens the fine-tuned model scores 20.5,
against 22.4 zero-shot and 20.6 for the original model.

The fairest one-line summary is the comparison with the model as it was trained. The original Model B reads the first
1,024 tokens of these pages at a perplexity of 20.6, and can't read further. The fine-tuned model reads all 4,096 at
21.0, and its loss at positions 2,048–4,096 (3.05) is within 0.03 of the original model's at 512–1,024. Four times
the context, at very nearly the same per-token quality, for 1% of the original training.

Two cautions keep this honest. First, the fine-tune trained on exactly this kind of text, the opening 4,096 tokens of
long pages, so some of its gain is plain adaptation to the page type: it's better than the original even at the
first 256 tokens (21.1 against 21.4), where position plays no part. Second, the per-position loss stays level rather
than falling past 1,024, for the fine-tuned model as for the original one within its own range. Later tokens in these
pages aren't easier to predict, so a level curve is what a model reading the whole page should show, but it can't
prove how much of the distant context the model actually draws on. That takes a sharper test: score the same late
positions with the context cut back to 1,024 tokens, and see how much worse they get.

![Perplexity at 1,024 and 4,096 tokens against fine-tuning steps](/assets/s1/positions/figures/yarn_finetune_steps.png)

**How to read it.** Each point is one snapshot of the fine-tune, scored on the same 500 pages; "zero-shot" is YaRN
before any training. The teal line is the perplexity of the whole 4,096-token page; the gold line counts only its
first 1,024 tokens, the length the model was originally trained on. Both come from the same pass over the same pages.
The dashed line is Model B before YaRN on those same first 1,024 tokens: gold above it means stretching has cost the
model quality inside its old range, gold below it means that cost is gone.

Most of the gain comes immediately. The first 25 steps (12M tokens, about 0.1% of pretraining) take whole-page
perplexity from 23.2 to 21.5, 77% of the total gain (1.7 of 2.2); by step 100 it's 21.1 (94%), and the last 100
steps add almost nothing. The gold line is back at the original model's quality by step 75. Of the two anchors
carried over from the paper, the share of pretraining turned out to be the better guide to where the curve bends.

## 7. What broke (Part 2)

Part 1 has the training-side failures. A few belonged to this part's work:

- **The KV cache, built by hand, had its own bugs before it matched full recomputation:**
  - a position embedding off by one;
  - the cache length read with `len()` instead of the sequence dimension;
  - key positions computed from the new tokens only;
  - the subtle one: PyTorch's `is_causal` aligning the mask top-left when there are fewer queries than keys.

  The cached-vs-recomputed test from Section 2 caught every one.
- **The H100 benchmark first ran on the CPU.** The device helper checked for Apple's GPU and otherwise fell back to the
  CPU, never checking for CUDA. On the Mac that was invisible; on the pod, four idle H100s and 90 busy CPU cores gave
  it away.
- **A formula that double-counted.** The KV-cache size formula multiplied by the number of stored positions twice,
  after a refactor. It was caught by comparing the formula against the measured cache sizes, the same check the
  memory figure makes.
- **The fine-tune's first draft copied "resume".** Continuing a finished run looks like resuming, so the first draft
  reused the resume path wholesale: the old context length, the old loader position (which the long-page loader
  rejected), the old best validation loss and the old experiment-tracking run. A laptop dry run caught all four
  before any GPU time.
- **`is` instead of `==`.** A check written `init_from is "resume"` compares object identity, not text. It happened
  to work for the value set in a config file and would have failed silently for the same word typed on the command
  line.
- **Parity in the wrong place.** The fine-tune evaluated as often as pretraining did, every 5% of the run, but each
  evaluation was still pretraining-sized: about 12M tokens scored every 10 steps, more than the 4.9M trained in
  between. Evaluations and saves took about 70% of the 23-minute run, and because every logged step was an evaluation
  step, MFU never got logged. Fixed: evaluate only at snapshots, on fewer batches.
- **A dry run that started downloading the dataset.** The first laptop dry run went through the training script's
  normal data setup, which fetches every missing shard, and had pulled about 13 GB before it was stopped. Dry runs now
  stub the download out.

## 8. Limitations and what carries over

The inference numbers come from one model size, two machines and float32 without kernel fusion or CUDA graphs, all of
which push a 124M model towards being overhead-bound. The position analysis covers one context length. Here's what
should carry over to larger models and longer contexts, and what each expectation rests on:

| Finding at 124M | With bigger models | With longer contexts | Basis |
|---|---|---|---|
| KV cache 6.4× smaller | Same ratio per layer; the absolute size grows with d·L | The ratio grows (≈19× at 32k for a 7B-class model with Gemma 3's layout) | Arithmetic |
| Decode speed depends on the machine; overhead-bound on H100 | Becomes memory-bound (a 7B model's weights alone take ~4 ms to read per step), so the smaller cache pays | Cache traffic dominates | Arithmetic + both measurements |
| Reading the cache: 1 FLOP per byte for A, 4 for B | Unchanged: it depends on n_h/n_kv, not size | Unchanged, but the cache grows | Arithmetic |
| RoPE with base 10,000; 18 of 32 bands turn within the context | — | The base must rise (Llama 3: 500k; Gemma 3 global: 1M), or bands must be rescaled (YaRN) | Arithmetic + papers [[3]](#ref-3), [[9]](#ref-9), [[10]](#ref-10) |
| YaRN at 4×: zero-shot removes the breakdown; 25 fine-tuning steps (~0.1% of pretraining) give most of the gain | The paper's own results are at 7B and above; the 0.1% budget matched here too | Larger stretches need more fine-tuning, or the base raised in pretraining | One model, one factor, one seed |
| Model B's induction heads sit in a local layer, so it can't copy from 768 tokens back | Unknown: larger hybrid models may place copying heads in global layers | The window caps copying unless global layers learn it | One seed per model; a 64-token block |
| A sinks attention on position 0, B (RoPE) on separator tokens | Attention sinks are widespread at scale (Xiao et al., 2023) [[6]](#ref-6) | Sinks on a token need that token within reach of local layers | 32 windows, both models |

A second seed of Model B could, for instance, place its induction head in a global layer, which would change the
copying result. The attention analysis is also descriptive: it shows where attention goes, not that the model needs
it there. A causal check, switching a head off and measuring what breaks, is the natural next step.

## Appendix

### What the KV cache has to get right

The details behind Section 2's KV cache, for every rung of the ladder:

- **Positions.** A new token at position 700 needs position 700's embedding (Model A) or rotation (Model B), not
  position 0's. The cache keeps a running position counter.
- **RoPE.** Keys are cached *after* rotation, so each cached key keeps the position it was rotated at.
- **GQA.** Only the 3 key/value heads go into the cache, not all 12 query heads. That's where the 4× saving comes
  from.
- **Local windows.** A local layer never looks further back than its window, so it can throw away anything older.
  Its cache stops growing at 511 positions.
- **The causal mask.** PyTorch's `is_causal=True` lines the mask up from the top-left corner. That's correct when
  queries and keys have the same length, and wrong when one new query faces 700 cached keys. The cached path builds
  its mask explicitly from the true positions instead.

### Attention, FlashAttention and the quadratic

The worked version of Section 3's answer.

All of this has been counted per token, and per token, attention is *linear* in context length: a token at position p
compares itself with p + 1 keys, so it costs 4d·(p + 1) FLOPs. That's why the formula uses the average, 4d·k̄. The
famous quadratic appears when you add up a whole sequence. T tokens, each attending to everything before it, make
T(T + 1)/2 query-key pairs, about 2d·T² FLOPs per layer. Both statements are true; they just count different things.
For the 7B-class model with full attention, the attention FLOPs for one sequence grow from about 0.3 TFLOP at 1k
tokens to about 4,500 TFLOP at 128k: 128 times the context, 16,000 times the work.

FlashAttention (Dao et al., 2022) [[11]](#ref-11) doesn't change that arithmetic: it computes exact attention, and its backward pass
even recomputes the scores instead of storing them. What it changes is memory traffic. Instead of writing the T × T
score matrix to GPU memory and reading it back twice, it works in tiles that fit on-chip and fuses score, softmax and
weighted sum into one kernel. Memory grows with T instead of T², and the time saved is time spent waiting on memory,
the memory-bandwidth regime from Section 2.

Our own runs show why that matters even at 1,024 tokens. With 12 heads and 60 sequences per GPU, the score matrices
for one layer would take about 1.5 GB in bf16, and keeping all 12 layers' for the backward pass would take about
18 GB. At 128k tokens, one head's scores for one sequence would be 34 GB, more than a third of an H100's memory.
Without FlashAttention, long-context training simply wouldn't fit.

So FlashAttention makes quadratic attention practical; it doesn't make it linear. Within exact attention, only
restricting which keys a query sees changes the order itself, as Model B's local windows do: a local layer costs
O(T·W) per sequence, linear in T. In time, that saving needs a kernel that skips the blocks outside the window, like
FlexAttention's block masks or FlashAttention's window option. With an ordinary dense mask, all T² scores are
computed and most thrown away, which is why the dense-mask version in Part 1's ladder trained *slower*; FlexAttention recovered about 5% of that.

Decoding with a KV cache is the same picture seen one step at a time. Each new token costs O(T), one pass over the
cache, so generating a whole sequence still adds up to O(T²), just spread across the steps.

### How the band measures are computed

The two measures behind Section 4's band-usage heatmap and its three real heads.

**From the weights.** Write W_q^{(i)} for the rows of the head's query projection
that produce band i's two dimensions, and W_k^{(i)} for the same rows of the key projection it shares under
grouped-query attention, each with the RMSNorm gain in front of attention folded in. The head's **q·k weight** in band
i is the product of their sizes, as a share of the head's total; its **slow-band weight** sums that share over the 14
slow bands, and a head that treated every band alike would get 14/32, or 44%:

$$w_i = \big\lVert W_q^{(i)} \big\rVert \cdot \big\lVert W_k^{(i)} \big\rVert, \qquad
s_i = \frac{w_i}{\sum_{j=1}^{32} w_j}, \qquad
\text{slow-band weight} = \sum_{i \,\in\, \text{slow}} s_i$$

where the norms are Frobenius norms.

**On real text.** Using the complex form from Section 4's derivation, write z_i^q(m) = q_i(m) + i·q_{i+h/2}(m) for band
i of the query at position m, *before* RoPE's rotation, and z_i^k(n) likewise for the key. Band i's average contribution
to the score at distance Δ, over a window of T = 1,024 tokens, is

$$S_i(\Delta) = \frac{1}{T - \Delta} \sum_{n=0}^{T-1-\Delta}
\frac{1}{\sqrt{h}}\, \mathrm{Re}\Big[\, z_i^q(n+\Delta)\; \overline{z_i^k(n)}\; e^{\,i\,\Delta\,\theta_i} \Big]$$

averaged over 64 held-out windows. The factor e^{iΔθ_i} is RoPE's rotation at that distance; everything else is
content. The figure's bottom row is the head's mean score, S(Δ) = Σ_i S_i(Δ). The top row sums S_i(Δ) over each speed
group: fast, middle and slow. (The averaging over n is a cross-correlation, so it takes one FFT per head rather than
a T × T loop.)

### Reproducing this part

The models are public on Hugging Face: [Model A](https://huggingface.co/CuriosLM/gpt2-124m-fineweb-edu),
[Model B](https://huggingface.co/CuriosLM/gemma-style-113m-fineweb-edu) and the
[YaRN-extended Model B](https://huggingface.co/CuriosLM/gemma-style-113m-fineweb-edu-yarn4k) (4,096-token context, after the 200-step
fine-tune). The code is in the [CuriosLM repository](https://github.com/Shinibali/CuriosLM):

- **Inference:** `analysis/efficiency/`:
  - `01_inference_benchmark.py` (`--tag h100` for another machine);
  - `03_decode_bytes.py`, `04_devices.py`;
  - `05_flops_scaling.py` and `06_memory_scaling.py`, for the long-context FLOPs and memory arithmetic.
- **Positions:** `analysis/positions/` (`01_rope_bands.py`, `02_learned_positions.py`, `03_band_usage.py`,
  `04_real_heads.py`, `05_rope_scaling.py`).
- **Attention:** the evaluation-only attention path in `curioslm/model.py` (`GPT.attention_weights`, tested in
  `tests/curioslm/test_attention_weights.py`) and `analysis/attention/`: `01_emoji_probe.py` (`--text-only` prints
  the continuations alone), `02_disagreement.py`, `03_attention_distance.py`, `04_induction.py`.
- **YaRN:** the scaled RoPE tables in `curioslm/model.py` (tested in `tests/curioslm/test_yarn.py`); the band and
  context figures, `analysis/positions/06_yarn_bands.py` and `07_yarn_context.py` (`--finetuned <out_dir>` scores
  the fine-tune's snapshots); the fine-tune itself, `train.py config/yarn_finetune.py` on one GPU, which continues
  Model B's run through `init_from='extend'` and reads only pages of 4,096+ tokens
  (`curioslm/loaders/long_page_loader.py`).
- **The KV cache:** `curioslm/model.py`, tested in `tests/curioslm/test_kv_cache.py`.

### Thanks

The thanks in Part 1 apply here too, and two in particular. Horace He's
[*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html) is the frame for Section
2's three bottlenecks. Aleksa Gordić's [*Inside the Transformer: The Life of a Token*](https://www.aleksagordic.com/blog/transformer)
is the best place to read about RoPE, YaRN and hybrid attention in depth.

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

### Citation

```bibtex
@misc{bhattacharyya2026gpt_to_gemma_2,
  author       = {Bhattacharyya, Shinibali},
  title        = {GPT-2 (124M) → Gemma 3-Style (113M), Part 2: KV Caches, Attention Visualization, RoPE Clocks and 4× Context Extension},
  year         = {2026},
  url          = {https://shinibali.github.io/gpt2-to-gemma-part-2/},
}
```
