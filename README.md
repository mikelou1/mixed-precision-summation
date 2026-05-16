# Beneath the Surface of Mixed-Precision Summation in High Performance Computing
### Tomorrow's Accuracy at Yesterday's Cost

> **Score: 9992.26 / 10000** on the official benchmark.

## Abstract

Floating-point summation is the primitive operation at the core of dot products, matrix multiplication, and numerical solvers, and its numerical behavior is consequential across virtually all of scientific computing. A single forward pass of a 70B-parameter transformer involves on the order of $10^{13}$ reduction operations; the choice of precision and ordering across those reductions determines both the final loss curve and the wall-clock cost of training. The IEEE 754 standard defines three binary formats relevant to this setting: fp64, fp32, and fp16, with unit roundoffs of approximately $10^{-16}$, $10^{-7}$, and $10^{-3}$ respectively, ordered by decreasing numerical fidelity and decreasing memory and compute cost. fp64 is the historical default for numerical stability, but its 64-bit operand width limits parallel occupancy on contemporary hardware. fp16 delivers up to $4\times$ lower memory footprint and $16\times$ greater compute speed on NVIDIA tensor-core accelerators, but rounding error accumulates rapidly and after sufficiently many operations the result is unreliable. Neither format is adequate in isolation, and an optimal blueprint over the space of precision assignments and summation orderings is computationally intractable at the scale scientific workloads demand.

## Overview

- [The Problem](#the-problem)
- [Hardware Context](#hardware-context)
- [A Hardware-Aligned Efficiency Metric](#a-hardware-aligned-efficiency-metric)
- [Non-technical Explanation](#non-technical-explanation)
- [Evaluation of Different Approaches](#evaluation-of-different-approaches)
- [Reproduce Results](#reproduce-results)

## The Problem

Given 275,000 floating-point values drawn uniformly from $[0, 1)$, produce a blueprint that computes their sum. A blueprint specifies both the order of additions and the IEEE 754 precision used at each step, with each step using fp16, fp32, or fp64. Its output is the value produced by the final addition.

The objective is to maximize an efficiency metric, defined in the next section, that rewards accuracy and penalizes computational cost. The two are in direct tension. Performing all additions in fp64 produces a near-exact result but is roughly eight times slower than fp16. Performing all additions in fp16 is fast but accumulates rounding error so quickly that the final result is unrecognizably wrong. The interesting blueprints lie between these extremes, and the space of valid blueprints grows combinatorially with the input size.

Two specific failure modes shape every viable approach. Absorption error occurs when a large partial sum subsumes a small operand entirely. For instance, `1000 + 0.0001` in fp16 is just `1000`, with the small contribution lost. Non-associativity means that summation order itself determines accuracy: the same values summed left-to-right and summed pairwise can produce different results, with worst-case error scaling as $O(n\varepsilon)$ in the first case and only $O(\varepsilon \log n)$ in the second. Precision loss is also irreversible. A value rounded under fp16 cannot be recovered by performing subsequent operations in fp32 or fp64. These three facts together explain why the problem is non-trivial: a good blueprint cannot be derived from any single principle (sort the values, use a tree, switch to higher precision near the root) but must combine several.

## Hardware Context

The cost weights used in the efficiency metric below are not chosen for analytical convenience; they reflect measurable throughput ratios on current NVIDIA datacenter GPUs. The table below reports peak dense tensor-core throughput across the precision formats relevant to scientific computing, spanning the Ampere (A100), Hopper (H100), and Blackwell (B200) generations.

| GPU      | Year | FP64 TC | FP32/TF32 TC | FP16/BF16 TC | FP8 TC | HBM (GB) | BW (TB/s) |
|----------|------|---------|--------------|--------------|--------|----------|-----------|
| A100 SXM | 2020 | 19.5    | 156          | 312          | ...    | 80       | 2.04      |
| H100 SXM | 2022 | 67      | 989          | 1,979        | 3,958  | 80       | 3.35      |
| B200     | 2024 | 40      | 2,200        | 4,500        | 9,000  | 192      | 8.00      |

All throughput values are TFLOPS unless otherwise marked. Values gathered from NVIDIA datasheets and the Hopper architecture whitepaper.

![Hardware throughput chart](assets/hardware_chart.png)

Two trends in the table bear directly on the problem. First, fp16 tensor-core throughput consistently exceeds fp32 by a factor of roughly $2\times$ and fp64 by $8\times$ within a generation; the cost weights $1:2:8$ used in the metric are taken from this ratio. Second, peak throughput at low precision has grown roughly $15\times$ from A100 to B200, while fp64 has roughly doubled. The relative cost of double-precision arithmetic on tensor-core hardware is widening, not narrowing. Mixed-precision summation is therefore not an optimization of academic interest. It is the only path to using current hardware near its peak.

## A Hardware-Aligned Efficiency Metric

The metric is grounded in measurable hardware behavior rather than chosen for analytical convenience. Accuracy and cost are both functions of the same blueprint and move in opposition. No total ordering over blueprints exists by either variable alone, since a blueprint that is more accurate may be strictly more expensive and vice versa. Direct comparison between algorithms is therefore ill-defined without a means of collapsing the two dependent variables into a single quantity. $\mathcal{E} = \alpha \cdot \beta$ serves this purpose, with weights chosen so the ordering it induces corresponds directly to the efficiency ordering that would be observed on the hardware above.

**Accuracy factor.** Let $S$ be the value computed by a blueprint and $\Sigma$ the exact sum. The relative error is

$$\eta = \frac{|S - \Sigma|}{\Sigma + \tau}$$

where $\tau$ is a small stabilising constant. The accuracy factor is

$$\alpha = \min\!\left(1,\ \max\!\left(0,\ \frac{-\log_2 \eta}{24}\right)\right)$$

normalised against the 24-bit mantissa of fp32, which is the practical accuracy ceiling for any blueprint operating below fp64. The $\log_2$ scaling is dictated by the IEEE 754 mantissa structure: a blueprint that retains $k$ effective bits has $\eta \approx 2^{-k}$, so a linear function of $-\log_2 \eta$ corresponds linearly to retained precision. A bit-based formulation is necessary because the error regime of fp16-based summations, typically $10^{-6}$ to $10^{-3}$, causes a linear accuracy term to saturate and fail to discriminate between blueprints.

**Why precision above 24 bits is not rewarded.** The clamp $\alpha = \min(1, \cdot)$ at $\eta = 2^{-24}$ is not arbitrary. The cap reflects the practical accuracy ceiling: any blueprint that incurs even one fp32 add cannot, in general, exceed fp32 mantissa precision, since each fp32 cast quantises its operand to 24 bits. A blueprint producing $\eta = 10^{-50}$ is no more useful in a mixed-precision setting than one producing $\eta = 2^{-24}$; both deliver inputs of identical quality to any downstream fp32 computation. The cap also prevents pathological metric values that would otherwise let an all-fp64 blueprint dominate the rank order through accuracy alone. By capping at the fp32 mantissa width, the metric forces algorithms to compete on the operationally meaningful axis: cost to reach the practical accuracy ceiling, not raw error magnitude. This is precisely the optimisation problem mixed-precision GEMM, FlashAttention, and similar production kernels actually solve.

A corollary is that the optimisation problem is properly stated as a constrained minimisation: minimise $C$ subject to $\eta \le 2^{-24}$. Blueprints that drive $\eta$ below this threshold pay additional cost for no marginal reward and are therefore suboptimal by construction. A well-designed solver calibrates its precision allocation to land $\eta$ just below the threshold and stops.

**Cost factor.** Let $a_{16}$, $a_{32}$, $a_{64}$ be the number of additions at each precision. The weighted cost is

$$C = a_{16} + 2\,a_{32} + 8\,a_{64}$$

with weights $1:2:8$ reflecting the tensor-core throughput ratios in the hardware table. The weight of $8$ for fp64 ensures it is dominated by fp32 on a per-add basis, since fp64 cannot improve $\alpha$ beyond the fp32 ceiling. The cost factor is

$$\beta = \frac{n - 1}{C}$$

which equals $1$ for an all-fp16 blueprint and decreases as higher-precision additions are introduced.

**Composite metric.**

$$\mathcal{E} = \alpha \cdot \beta, \qquad \mathcal{E} \in [0,\,1]$$

Blueprints approaching $\mathcal{E} = 1$ simultaneously achieve fp32-grade accuracy and fp16-grade cost, which is the operating point mixed-precision scientific computation targets in practice.

## Non-technical Explanation

Computers represent decimal numbers using a fixed number of digits. The more digits, the more accurate the representation, but also the more memory each number occupies, and the more time each arithmetic operation takes. Three standard formats are in widespread use today, and they sit at very different points on this tradeoff. The largest format, fp64, holds about 16 digits and is the historical default for any computation where accuracy matters. The middle format, fp32, holds about 7 digits at half the cost. The smallest format, fp16, holds only 3 digits but runs roughly eight times faster than fp64 on modern GPUs.

In a single addition between two values stored at the same format, the result is rounded back to that format's digit count. Adding two fp16 values produces an fp16 value with 3 digits of accuracy; adding two fp64 values produces an fp64 value with 16. The choice of format is therefore not just about how individual numbers are stored. It determines how much accuracy survives every operation.

This problem asks how to sum a list of 275,000 numbers under that constraint. Adding them all in fp64 is accurate but slow. Adding them all in fp16 is fast but disastrously inaccurate: the running total rapidly exceeds the range where fp16's three digits can distinguish small contributions from one another, and most of the inputs simply disappear into the rounding. Neither extreme is acceptable.

The answer is to mix the formats. Most of the additions can happen in fp16 with negligible error, provided the values being added are similar in magnitude to one another. The handful of additions where errors would be largest, typically the last few, where partial sums have grown large and small contributions risk being lost, can be promoted to fp32 or fp64. The blueprint specifies which precision to use at each step and in what order the additions occur. A well-designed blueprint achieves the accuracy of a pure fp32 computation at close to the cost of a pure fp16 one.

The animation below shows a small blueprint executing on four input values. The pairs `(0.7856, 0.4204)` and `(0.1843, 0.9117)` are each summed in fp16, after which the two results are summed in fp32. The red zeros mark digits that fp16 cannot represent. They appear in the result not because the calculation produced them, but because the next stage's fp32 adder expects seven digits of input and the fp16 stage only gave it three of meaning. The trailing zeros are treated as genuine input by the fp32 adder, which is why the final result `2.290000` differs from the exact value `2.302023` by more than fp32 alone would produce.

![Mixed-precision evaluation animation](assets/evaluation.gif)

## Evaluation of Different Approaches

A series of progressively stronger blueprints were evaluated against the benchmark. Each row reports the total score across all 100 cases out of a maximum of 10,000.

[Soon to be written]: #

## Reproduce Results

All files necessary to reproduce the score are included in this repository:

- `s.py`, the solver implementing `solve(n, values)`
- `judge.cpp`, IEEE 754 simulator that evaluates a blueprint and returns its score
- `gen.py`, deterministic generator for the 100 benchmark cases
- `g.py`, multi-process grader that runs the solver against all cases and reports the total

The only requirements are Python 3 and a C++17 compiler. No third-party packages are needed; everything uses the standard library.

To produce the benchmark and grade the solver:

```bash
g++ -std=c++17 -O2 -o judge judge.cpp
python3 gen.py
python3 g.py s.py
```

The grader runs cases in parallel across all available cores and uses `curses` for its progress display, so a Unix-like terminal (macOS or Linux) is recommended. Wall-clock runtime on a typical multi-core machine is roughly one to two minutes. The total score is printed to stdout on exit.
