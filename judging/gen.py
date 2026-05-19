import os
import random
from decimal import Decimal, getcontext

getcontext().prec = 60
SEED, NUM_CASES, N = 667676767, 100, 275000
TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
SAMPLE_CASES = {
    'sample1': [0.1, 0.2, 0.0001, 0.9999],
    'sample2': [0.9, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003],
}
def write_case(name, values):
    sigma = sum(Decimal(repr(v)) for v in values)
    with open(os.path.join(TESTS_DIR, f"{name}.in"), "w") as f:
        f.write(f"{len(values)}\n")
        f.write(" ".join(repr(v) for v in values) + "\n")
    with open(os.path.join(TESTS_DIR, f"{name}.out"), "w") as f:
        f.write(str(sigma) + "\n")

os.makedirs(TESTS_DIR, exist_ok=True)
for name, values in SAMPLE_CASES.items():
    write_case(name, values)
    print(f"Generated {name}")
rng = random.Random(SEED)
for i in range(1, NUM_CASES + 1):
    values = [rng.random() for _ in range(N)]
    write_case(i, values)
    print(f"Generated {i:03d}/{NUM_CASES}", end="\r", flush=True)
print(f"\nDone. {NUM_CASES} test cases + {len(SAMPLE_CASES)} sample cases written to {TESTS_DIR}/")
