You have been provided with:

| File          | Role                                                                                |
|---------------|-------------------------------------------------------------------------------------|
| `statement.md` | Full problem description, scoring metric, and IEEE 754 simulation semantics.        |
| `gen.py`      | Deterministic generator that produces the 100 benchmark cases into `cases/`.        |
| `judge.cpp`   | IEEE 754 simulator that evaluates one blueprint against one case and prints its per-case score. |
| `g.py`        | Multi-process grader that runs your solver against all 100 cases and reports the total. |

Write `s.py` to solve the problem written in statement.md, with a Python module exposing:

```python
def solve(n: int, values: list[float]) -> str:
    # your code here
```

## Constraints

- **No hardcoded outputs, lookup tables, or seed-dependent branching.** The blueprint must be computed from the input values.
- **You must not exceed 3000ms time limit. Due to hardware restrictions, it is recommanded your program runs in log-linear or better.**
- **Standard library only, numpy is allowed.**
- **If you get submit a non-valid piece of code initially, you will only have 1 more chance to improve it**

## Process

You will be evaluated in two phases:

**Phase 1: Initial submission.** Build and reason about `s.py` however you wish: read the harness source, run it locally against the benchmark, inspect intermediate cases, study which case distributions appear, profile your solver, etc. Submit your initial `s.py` when ready. You will be given its total score and a per-case breakdown.

**Phase 2: Single revision.** Based on the feedback from Phase 1, you may submit one revised `s.py`. The final score will be the highest of the two.

Submit your `s.py` when ready.
