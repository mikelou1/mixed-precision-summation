# Note that since I don't own a Windows or Linux based device, I am not it will work on non-macOS systems

import sys
import os
import time
import importlib.util
import multiprocessing
import random
import math
import subprocess

try:
    import curses
except ImportError:
    if sys.platform == 'win32':
        sys.stderr.write(
            "Error: curses is not available. On Windows, install windows-curses:\n"
            "    pip install windows-curses\n"
        )
    else:
        sys.stderr.write("Error: curses is not available on this system.\n")
    sys.exit(1)

from decimal import Decimal, getcontext
getcontext().prec = 60

SEED, NUM_CASES, N, TIME_LIMIT_MS, MEMORY_LIMIT_MIB, MAX_SCORE_PER_CASE = 667676767, 100, 275000, 3000, 1024, 100.0
TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
JUDGE_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge.exe" if sys.platform == 'win32' else "judge")
SAMPLES = ['sample1', 'sample2']

def generate_case(case_idx):
    rng = random.Random(SEED)
    for _ in range(case_idx - 1):
        for _ in range(N):
            rng.random()
    return [rng.random() for _ in range(N)]

def exact_sum(values):
    return sum(Decimal(repr(v)) for v in values)

def load_case(case_idx):
    in_path  = os.path.join(TESTS_DIR, f"{case_idx}.in")
    out_path = os.path.join(TESTS_DIR, f"{case_idx}.out")
    if os.path.exists(in_path) and os.path.exists(out_path):
        with open(in_path) as f:
            n = int(f.readline().strip())
            values = list(map(float, f.readline().split()))
        with open(out_path) as f:
            sigma = f.readline().strip()
        return values, sigma
    if isinstance(case_idx, int):
        values = generate_case(case_idx)
        sigma = str(exact_sum(values))
        return values, sigma
    raise FileNotFoundError(f"Test file not found: {in_path}")

_solve_fn = None

def _worker_init(solve_path):
    global _solve_fn
    spec = importlib.util.spec_from_file_location("solver", solve_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _solve_fn = mod.solve

def worker(args):
    solve_path, case_idx = args

    if isinstance(case_idx, int) and case_idx < 0:
        return {'case': case_idx, 'score': 0.0, 'time_ms': 0, 'memory_mib': 0, 'error_msg': None}

    result = {'case': case_idx, 'score': 0.0, 'time_ms': None, 'memory_mib': None, 'error_msg': None}

    try:
        values, sigma = load_case(case_idx)
        n = len(values)
    except Exception as e:
        result['error_msg'] = f"Load error: {e}"
        return result

    if _solve_fn is None:
        result['error_msg'] = "Solver not loaded"
        return result

    mem_before = 0
    try:
        if sys.platform != 'win32':
            import resource
            mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 if sys.platform == 'darwin' else 1)
    except Exception:
        mem_before = 0

    t0 = time.perf_counter()
    try:
        schedule_str = _solve_fn(n, list(values))
    except Exception as e:
        result['time_ms'] = int((time.perf_counter() - t0) * 1000)
        result['error_msg'] = f"Runtime error: {str(e)[:60]}"
        return result
    elapsed_ms = (time.perf_counter() - t0) * 1000
    result['time_ms'] = int(elapsed_ms)

    if elapsed_ms > TIME_LIMIT_MS:
        result['error_msg'] = f"TLE ({elapsed_ms:.0f}ms)"
        return result

    try:
        if sys.platform == 'win32':
            result['memory_mib'] = 0
        else:
            import resource
            mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 if sys.platform == 'darwin' else 1)
            result['memory_mib'] = max(0, int((mem_after - mem_before) / 1024))
    except Exception:
        result['memory_mib'] = 0

    if result['memory_mib'] > MEMORY_LIMIT_MIB:
        result['error_msg'] = f"MLE ({result['memory_mib']}MiB)"
        return result

    try:
        proc = subprocess.run(
            [JUDGE_BIN, str(case_idx), TESTS_DIR],
            input=schedule_str + "\n",
            capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            msg = proc.stderr.strip().replace('\n', ' ')[:80]
            result['error_msg'] = f"Invalid: {msg}"
            return result
        result['score'] = float(proc.stdout.strip().split()[0])
    except subprocess.TimeoutExpired:
        result['error_msg'] = "Judge timeout"
    except Exception as e:
        result['error_msg'] = f"Judge error: {e}"

    return result

def fmt_score(s):
    return f"{s:08.5f}"

def fmt_time(ms):
    if ms is None: return "  --"
    return f"{ms:4d}ms"

def fmt_memory(mib):
    if mib is None: return "  --"
    return f"{mib:4d}MiB"

def run_with_curses(stdscr, solve_path):
    curses.curs_set(0)
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    curses.init_pair(1, curses.COLOR_GREEN,  bg)
    curses.init_pair(2, curses.COLOR_RED,    bg)
    curses.init_pair(3, curses.COLOR_YELLOW, bg)
    curses.init_pair(4, curses.COLOR_WHITE,  bg)
    curses.init_pair(5, curses.COLOR_CYAN,   bg)

    height, width = stdscr.getmaxyx()
    COLS   = 5
    ROWS   = math.ceil(NUM_CASES / COLS)
    CELL_W = width // COLS

    states  = {k: {'score': None, 'time_ms': None, 'memory_mib': None, 'error_msg': None, 'done': False}
               for k in SAMPLES + list(range(1, NUM_CASES + 1))}
    running = set()

    tests_exist = os.path.exists(os.path.join(TESTS_DIR, "1.in"))
    header = (
        f"Grading: {os.path.basename(solve_path)} | "
        f"Seed: {SEED} | Cases: {NUM_CASES} | n: {N} | "
        f"TL: {TIME_LIMIT_MS}ms | ML: {MEMORY_LIMIT_MIB}MiB | "
        f"{'tests/ found' if tests_exist else 'generating on the fly'}"
    )

    def draw_header(total):
        done = sum(1 for k, s in states.items() if s['done'] and isinstance(k, int))
        try:
            stdscr.addstr(0, 0, header[:width-1], curses.A_BOLD)
            stdscr.addstr(1, 0, f"Progress: {done:3d}/{NUM_CASES}   Total: {total:010.5f} / {NUM_CASES * MAX_SCORE_PER_CASE:.5f}"[:width-1])
        except curses.error:
            pass

    def draw_sample(key):
        s = states[key]
        idx  = SAMPLES.index(key)
        col  = idx * CELL_W
        label = f"S{idx+1:02d}"
        if s['done']:
            text = f"[{label}] {fmt_score(s['score'])} {fmt_time(s['time_ms'])} {fmt_memory(s['memory_mib'])}"
            attr = curses.color_pair(2) if s['error_msg'] else curses.color_pair(5)
        elif key in running:
            text = f"[{label}] 00.00000 ---- ----"
            attr = curses.color_pair(3)
        else:
            text = f"[{label}] --.----- ---- ----"
            attr = curses.color_pair(4)
        try:
            stdscr.addstr(3, col, text[:CELL_W-1].ljust(CELL_W-1), attr)
        except curses.error:
            pass

    def draw_case(i):
        s   = states[i]
        row = 5 + ((i - 1) // COLS)
        col = ((i - 1) % COLS) * CELL_W
        if row >= height - 2:
            return
        if s['done']:
            text = f"[{i:03d}] {fmt_score(s['score'])} {fmt_time(s['time_ms'])} {fmt_memory(s['memory_mib'])}"
            attr = curses.color_pair(2) if s['error_msg'] else curses.color_pair(1)
        elif i in running:
            text = f"[{i:03d}] 00.00000 ---- ----"
            attr = curses.color_pair(3)
        else:
            text = f"[{i:03d}] --.----- ---- ----"
            attr = curses.color_pair(4)
        try:
            stdscr.addstr(row, col, text[:CELL_W-1].ljust(CELL_W-1), attr)
        except curses.error:
            pass

    def draw_errors():
        error_row = 5 + ROWS + 1
        errors = [(k, s['error_msg']) for k, s in states.items() if s['done'] and s['error_msg']]
        if not errors:
            return
        try:
            stdscr.addstr(error_row - 1, 0, f"Errors: ({len(errors)})"[:width-1], curses.A_BOLD)
        except curses.error:
            pass
        for idx, (k, msg) in enumerate(errors[:10]):
            line = error_row + idx
            if line >= height - 1:
                break
            label = f"S{SAMPLES.index(k)+1:02d}" if isinstance(k, str) else f"{k:03d}"
            try:
                stdscr.addstr(line, 0, f"  [{label}] {msg}"[:width-1], curses.color_pair(2))
            except curses.error:
                pass
        if len(errors) > 10:
            try:
                stdscr.addstr(error_row + 10, 0, f"  ... and {len(errors)-10} more"[:width-1], curses.color_pair(2))
            except curses.error:
                pass

    stdscr.clear()
    draw_header(0.0)
    for s in SAMPLES:
        draw_sample(s)
    for i in range(1, NUM_CASES + 1):
        draw_case(i)
    stdscr.refresh()

    total_score = 0.0

    with multiprocessing.Pool(
        processes=os.cpu_count(),
        initializer=_worker_init,
        initargs=(solve_path,)
    ) as pool:
        ncpu    = os.cpu_count()
        warmup  = [pool.apply_async(worker, ((solve_path, -i),)) for i in range(1, ncpu + 1)]
        for w in warmup:
            try: w.get()
            except: pass

        async_results = {}
        for s in SAMPLES:
            async_results[s] = pool.apply_async(worker, ((solve_path, s),))
            running.add(s)
            draw_sample(s)
        for i in range(1, NUM_CASES + 1):
            async_results[i] = pool.apply_async(worker, ((solve_path, i),))
            running.add(i)
            draw_case(i)
        stdscr.refresh()

        total_keys = len(SAMPLES) + NUM_CASES
        done_count = 0
        while done_count < total_keys:
            changed = False
            for k in list(async_results.keys()):
                if async_results[k].ready():
                    try:
                        r = async_results[k].get()
                    except Exception as e:
                        r = {'case': k, 'score': 0.0, 'time_ms': None, 'memory_mib': None, 'error_msg': f"Worker crash: {e}"}
                    states[k].update({**r, 'done': True})
                    running.discard(k)
                    if isinstance(k, int):
                        total_score += r['score']
                    del async_results[k]
                    done_count += 1
                    if isinstance(k, str):
                        draw_sample(k)
                    else:
                        draw_case(k)
                    changed = True

            if changed:
                draw_header(total_score)
                draw_errors()
                stdscr.refresh()
            time.sleep(0.02)

    draw_header(total_score)
    draw_errors()
    error_count = sum(1 for s in states.values() if s['error_msg'])
    final_row   = 5 + ROWS + 1 + error_count + 1
    try:
        stdscr.addstr(final_row, 0, f"Total: {total_score:010.5f} / {NUM_CASES * MAX_SCORE_PER_CASE:.5f}"[:width-1], curses.A_BOLD)
        stdscr.addstr(height - 1, 0, "Press any key to exit..."[:width-1])
    except curses.error:
        pass
    stdscr.refresh()
    stdscr.getch()
    return total_score

def main():
    if len(sys.argv) != 2:
        print("Usage: python g.py file.py")
        sys.exit(1)

    solve_path = os.path.abspath(sys.argv[1])

    if not os.path.exists(solve_path):
        print(f"Error: {solve_path} not found")
        sys.exit(1)

    if not os.path.exists(JUDGE_BIN):
        print(f"Error: judge binary not found at {JUDGE_BIN}")
        if sys.platform == 'win32':
            print("Build it with: g++ -std=c++17 -O2 -o judge.exe judge.cpp")
        else:
            print("Build it with: g++ -std=c++17 -O2 -o judge judge.cpp")
        sys.exit(1)

    if sys.platform == 'darwin':
        try:
            multiprocessing.set_start_method('fork')
        except RuntimeError:
            pass

    total = curses.wrapper(run_with_curses, solve_path)
    print(f"\nTotal: {total:010.5f} / {NUM_CASES * MAX_SCORE_PER_CASE:.5f}")

if __name__ == '__main__':
    main()
