# Mixed-Precision Summation Scheduling — Problem Statement

## Problem

Given a sequence of $n$ floating-point values, construct a summation schedule that maximises the efficiency metric $\mathcal{E} = \alpha \cdot \beta$, which jointly measures numerical accuracy and computational cost.

A summation schedule is a full binary tree in which each leaf holds one input value and each internal node carries a precision label $p \in \{\text{fp16}, \text{fp32}, \text{fp64}\}$. Values are accumulated left-to-right at each node using round-to-nearest-even. The output of the schedule is the value at the root.

## Function Signature

Your solution must be a Python file implementing:

```python
def solve(n: int, values: list[float]) -> str:
```

`values` is a list of $n$ IEEE 754 fp64 values in $[0, 1)$. Return a summation schedule as a nested expression where each group is:

```
(precision v1 v2 ... vk)
```

`precision` is one of `fp16`, `fp32`, `fp64`. Each `vi` is either a 1-indexed integer referring to an input value or a nested group. Groups are evaluated left-to-right.

## Constraints

- $n = 275{,}000$ for all test cases
- Every index from $1$ to $n$ must appear exactly once
- A malformed or invalid schedule scores $0$ for that case
- Time limit: $3000$ms per case (solve function only, grader overhead excluded)
- Memory limit: $1024$MiB

## Scoring

Accuracy and cost are both functions of the same schedule and move in opposition — a more accurate schedule may be strictly more expensive, and vice versa. Direct comparison is therefore ill-defined without collapsing the two into a single quantity. $\mathcal{E} = \alpha \cdot \beta$ serves this purpose, grounded in NVIDIA tensor-core hardware throughput ratios so the ordering it induces corresponds to real hardware efficiency.

Each of the 100 test cases is scored as:

$$\text{case score} = 100 \cdot \alpha \cdot \beta$$

**Accuracy factor.** Let $S$ be the computed value and $\Sigma$ the exact sum in extended precision:

$$\eta = \frac{|S - \Sigma|}{\Sigma + \tau}, \qquad \alpha = \min\!\left(1,\ \max\!\left(0,\ \frac{-\log_2 \eta}{24}\right)\right)$$

$\alpha = 1$ when the schedule retains at least 24 effective bits (the fp32 mantissa width) and decreases linearly with each lost binary digit.

**Cost factor.** Let $a_{16}$, $a_{32}$, $a_{64}$ be the number of two-operand additions at each precision:

$$C = a_{16} + 2\, a_{32} + 8\, a_{64}, \qquad \beta = \frac{n - 1}{C}$$

$\beta = 1$ for an all-fp16 schedule and decreases as higher-precision additions are introduced.

**Final score:**

$$\text{final score} = \sum_{i=1}^{100} \text{case score}_i \quad \in [0,\ 10{,}000]$$

Test cases are generated with seed $667676767$.

## Sample Cases

Sample cases are in `tests/sample1.in`, `tests/sample2.in` and corresponding `.out` files. They are judged but not counted toward the final score. Run them with:

```bash
python3 g.py --sample your_solution.py
```

---

### Sample 1

**Input** (`tests/sample1.in`)
```
4
0.1 0.2 0.0001 0.9999
```

| Value | IEEE 754 fp64 |
|-------|---------------|
| 0.1 | `3fb999999999999a` |
| 0.2 | `3fc999999999999a` |
| 0.0001 | `3f1a36e2eb1c432d` |
| 0.9999 | `3fefff2e48e8a71e` |

**Optimal output** (`tests/sample1.out` exact sum: `1.3000`)
```
(fp32 (fp32 1 2) (fp16 3 4))
```

Sum indices 3 and 4 first in fp16 — close in magnitude, minimal absorption error — then accumulate with 1 and 2 in fp32.

- $\eta \approx 3.668 \times 10^{-8}$, $\alpha = 1.000$, $\beta = 0.600$, **score = 60.00**

---

### Sample 2

**Input** (`tests/sample2.in`)
```
6
0.9 0.0003 0.0003 0.0003 0.0003 0.0003
```

| Value | IEEE 754 fp64 |
|-------|---------------|
| 0.9 | `3feccccccccccccd` |
| 0.0003 (×5) | `3f33a92a30553261` |

**Optimal output** (`tests/sample2.out` exact sum: `0.9015`)
```
(fp32 1 (fp16 2 3 4 5 6))
```

Sum the five small values first in fp16 — they are close in magnitude so absorption error between them is minimal — then add to index 1 in fp32 to avoid the large value absorbing the small sum. A naive all-fp16 left-to-right schedule absorbs all small values into 0.9 and scores only 41.92. An all-fp32 schedule avoids absorption but wastes cost, scoring only 50.00.

- $\eta \approx 9.405 \times 10^{-7}$, $\alpha = 0.834$, $\beta = 0.833$, **score = 69.51**

## Notes

- fp16 and fp32 arithmetic is simulated exactly per IEEE 754 in the judge
- The judge binary reads test files directly; only the schedule string is passed via stdin
- Build the judge: `g++ -std=c++17 -O2 -o judge judge.cpp`
- Generate test cases: `python3 gen.py`
- Grade a solution: `python3 g.py solution.py`
