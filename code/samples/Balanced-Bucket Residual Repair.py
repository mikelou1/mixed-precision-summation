import math
import struct
import heapq
import bisect

_F32 = struct.Struct('f')
_H16 = struct.Struct('<e')

def _f32(x):
    return _F32.unpack(_F32.pack(float(x)))[0]

def _h16(x):
    try:
        return _H16.unpack(_H16.pack(float(x)))[0]
    except OverflowError:
        return float('inf')

def _hsum(values, group, skip1=-1, skip2=-1):
    acc_set = False
    acc = 0.0
    for idx in group:
        if idx == skip1 or idx == skip2:
            continue
        v = _h16(values[idx])
        if not acc_set:
            acc = v
            acc_set = True
        else:
            acc = _h16(acc + v)
    return acc if acc_set else 0.0

def _f32_add_seq(base, values, raws):
    acc = _f32(base)
    for idx in raws:
        acc = _f32(acc + _f32(values[idx]))
    return acc

def _make_groups(values, order, q, small_target):
    # Weighted fair distribution over the sorted quantiles.  This keeps every
    # fp16 bucket with a similar value distribution instead of giving one bucket
    # only tiny numbers, which would freeze early in fp16.
    if small_target < 1e-9:
        small_target = 1e-9
    m = q + 1
    targets = [2048.0] * q + [float(small_target)]
    groups = [[] for _ in range(m)]
    sums = [0.0] * m
    heap = [(0.0, j) for j in range(m)]
    hp_pop = heapq.heappop
    hp_push = heapq.heappush
    vals = values
    for idx in order:
        _, j = hp_pop(heap)
        groups[j].append(idx)
        s = sums[j] + vals[idx]
        sums[j] = s
        hp_push(heap, (s / targets[j], j))
    small_out = _hsum(vals, groups[-1])
    return groups, small_out

def _find_one_raw(values, order, sorted_vals, owner, groups, base, target, delta):
    n = len(order)
    p = bisect.bisect_left(sorted_vals, delta)
    # Try a fairly wide window; for uniform random data this is far more than enough.
    best = None
    for radius in (64, 256, 1024, 4096):
        lo = max(0, p - radius)
        hi = min(n, p + radius + 1)
        cand = []
        for t in range(lo, hi):
            idx = order[t]
            g = owner[idx]
            if 0 <= g < len(groups) - 1:
                final = _f32_add_seq(base, values, (idx,))
                # Exact target is ideal, but within one fp32 bin around the target is also alpha=1.
                err = abs(final - target)
                if err <= 0.0078125:
                    cand.append((err, abs(values[idx] - delta), idx, g))
        cand.sort()
        for _, __, idx, g in cand[:80]:
            if _hsum(values, groups[g], idx) == 2048.0:
                return [idx]
        if cand:
            best = cand[0]
    return None

def _find_two_raw(values, order, sorted_vals, owner, groups, base, target, delta):
    # Two-pointer search for a pair of full-bucket leaves whose fp32 sum lands on target.
    full_order = [idx for idx in order if 0 <= owner[idx] < len(groups) - 1 and values[idx] <= delta]
    if len(full_order) < 2:
        return None
    l, r = 0, len(full_order) - 1
    trials = []
    vals = values
    while l < r and len(trials) < 400:
        i = full_order[l]
        j = full_order[r]
        s = vals[i] + vals[j]
        final = _f32_add_seq(base, vals, (i, j))
        err = abs(final - target)
        if err <= 0.0078125:
            trials.append((err, abs(s - delta), i, j))
            # collect nearby alternatives
            l += 1
            r -= 1
        elif s < delta:
            l += 1
        else:
            r -= 1
    trials.sort()
    for _, __, i, j in trials[:120]:
        gi = owner[i]
        gj = owner[j]
        if gi == gj:
            if _hsum(vals, groups[gi], i, j) == 2048.0:
                return [i, j]
        else:
            if _hsum(vals, groups[gi], i) == 2048.0 and _hsum(vals, groups[gj], j) == 2048.0:
                return [i, j]
    return None

def _find_three_raw(values, order, sorted_vals, owner, groups, base, target, delta):
    # Rare fallback. Pick one small candidate and reduce to two-sum.
    vals = values
    n = len(order)
    # try candidates around delta/3, also around a small value so pair has room
    anchors = [delta / 3.0, max(0.0, delta - 1.5), 0.5]
    tried = set()
    for a in anchors:
        p = bisect.bisect_left(sorted_vals, a)
        for t in range(max(0, p - 80), min(n, p + 81)):
            k = order[t]
            if k in tried:
                continue
            tried.add(k)
            gk = owner[k]
            if not (0 <= gk < len(groups) - 1):
                continue
            rem = delta - vals[k]
            if rem <= 0.0 or rem >= 2.0:
                continue
            base2 = _f32_add_seq(base, vals, (k,))
            # Build a restricted two-pointer ignoring k.
            full_order = [idx for idx in order if idx != k and 0 <= owner[idx] < len(groups) - 1 and vals[idx] <= rem]
            l, r = 0, len(full_order) - 1
            trials = []
            while l < r and len(trials) < 120:
                i = full_order[l]; j = full_order[r]
                s = vals[i] + vals[j]
                final = _f32_add_seq(base2, vals, (i, j))
                err = abs(final - target)
                if err <= 0.0078125:
                    trials.append((err, abs(s - rem), i, j))
                    l += 1; r -= 1
                elif s < rem:
                    l += 1
                else:
                    r -= 1
            trials.sort()
            for _, __, i, j in trials[:40]:
                # Check all affected groups after removals.
                by_g = {}
                for x in (k, i, j):
                    by_g.setdefault(owner[x], []).append(x)
                ok = True
                for g, xs in by_g.items():
                    if len(xs) == 1:
                        if _hsum(vals, groups[g], xs[0]) != 2048.0:
                            ok = False; break
                    else:
                        # len 2 or 3; hsum supports two skips, for 3 do manual filtered check
                        if len(xs) == 2:
                            out = _hsum(vals, groups[g], xs[0], xs[1])
                        else:
                            acc_set = False; acc = 0.0
                            skips = set(xs)
                            for idx in groups[g]:
                                if idx in skips: continue
                                v = _h16(vals[idx])
                                if not acc_set:
                                    acc = v; acc_set = True
                                else:
                                    acc = _h16(acc + v)
                            out = acc if acc_set else 0.0
                        if out != 2048.0:
                            ok = False; break
                if ok:
                    return [k, i, j]
    return None

def solve(n: int, values: list[float]) -> str:
    if n <= 1:
        return "1" if n == 1 else ""
    if n < 1000:
        return "(fp32 " + " ".join(str(i) for i in range(1, n + 1)) + ")"

    vals = values
    target = _f32(math.fsum(vals))
    q = int(target // 2048.0)
    if q < 1:
        return "(fp32 " + " ".join(str(i) for i in range(1, n + 1)) + ")"
    rem = target - 2048.0 * q

    order = sorted(range(n), key=vals.__getitem__)
    sorted_vals = [vals[i] for i in order]

    # The small fp16 bucket has a predictable upward drift when summed ascending.
    # Bias is close to rem/80 on these random cases; keep the window tight for speed.
    center = int(round(rem / 80.0))
    biases = set(range(0, 4))
    for d in range(-5, 6):
        b = center + d
        if b >= 0:
            biases.add(b)
    best = None
    # Limit the count so pathological large sets do not waste time.
    for b in sorted(biases):
        st = rem - float(b)
        if st <= 0.0:
            st = max(rem * 0.25, 1e-6)
        groups, small_out = _make_groups(vals, order, q, st)
        dlt = rem - small_out
        if dlt >= -1e-9:
            if best is None or dlt < best[0]:
                best = (dlt, groups, small_out)

    if best is None:
        # Last-ditch fallback: deliberately undershoot the remainder; raw leaves will repair it.
        groups, small_out = _make_groups(vals, order, q, max(1e-6, rem - max(8.0, rem / 60.0)))
        best = (rem - small_out, groups, small_out)

    delta, groups, small_out = best

    # Verify full buckets. If one undershot, choose a safer target distribution by lowering small target.
    full_outs = [_hsum(vals, g) for g in groups[:-1]]
    if any(o != 2048.0 for o in full_outs):
        # Conservative fallback: no small bucket target pressure; distribute into q full groups and a tiny small group.
        groups, small_out = _make_groups(vals, order, q, max(1e-6, rem - max(16.0, rem / 40.0)))
        full_outs = [_hsum(vals, g) for g in groups[:-1]]
        delta = rem - small_out

    # Parent grouping result for full chunks is exact if each child is 2048 and parent has <=31 children.
    # If the verifier fallback ever leaves a non-2048 chunk, use its real sum in the base estimate.
    base = sum(full_outs) + small_out

    owner = [-2] * n
    for gi, g in enumerate(groups):
        for idx in g:
            owner[idx] = gi

    raws = []
    if abs(_f32(base) - target) > 0.0078125:
        delta = target - _f32(base)
        if 0.0 <= delta < 1.0:
            raws = _find_one_raw(vals, order, sorted_vals, owner, groups, base, target, delta) or []
        elif 0.0 <= delta < 2.0:
            raws = _find_two_raw(vals, order, sorted_vals, owner, groups, base, target, delta) or []
        elif 0.0 <= delta < 3.0:
            raws = _find_three_raw(vals, order, sorted_vals, owner, groups, base, target, delta) or []

    # If exact landing failed, use the closest one-leaf correction that leaves its bucket stable.
    if not raws and abs(_f32(base) - target) > 0.0078125:
        delta = target - _f32(base)
        if delta > 0.0:
            p = bisect.bisect_left(sorted_vals, min(delta, 0.999999))
            candidates = []
            for radius in (512, 4096, 20000):
                lo = max(0, p - radius); hi = min(n, p + radius + 1)
                for t in range(lo, hi):
                    idx = order[t]; g = owner[idx]
                    if 0 <= g < len(groups) - 1:
                        final = _f32_add_seq(base, vals, (idx,))
                        candidates.append((abs(final - target), idx, g))
                candidates.sort()
                for _, idx, g in candidates[:100]:
                    if _hsum(vals, groups[g], idx) == 2048.0:
                        raws = [idx]
                        break
                if raws:
                    break

    rawset = set(raws)
    # Build fp16 chunk expressions.  Indices in the expression are 1-based.
    chunk_exprs = []
    for gi in range(q):
        arr = [str(idx + 1) for idx in groups[gi] if idx not in rawset]
        if len(arr) >= 2:
            chunk_exprs.append("(fp16 " + " ".join(arr) + ")")
        elif len(arr) == 1:
            chunk_exprs.append(arr[0])

    # Combine at most 31 chunks per fp16 parent, avoiding fp16 overflow at 32*2048.
    full_children = []
    for i in range(0, len(chunk_exprs), 31):
        block = chunk_exprs[i:i + 31]
        if len(block) == 1:
            full_children.append(block[0])
        else:
            full_children.append("(fp16 " + " ".join(block) + ")")

    small_arr = [str(idx + 1) for idx in groups[-1] if idx not in rawset]
    if len(small_arr) >= 2:
        small_expr = "(fp16 " + " ".join(small_arr) + ")"
    elif len(small_arr) == 1:
        small_expr = small_arr[0]
    else:
        small_expr = None

    root = full_children[:]
    if small_expr is not None:
        root.append(small_expr)
    root.extend(str(idx + 1) for idx in raws)

    if len(root) == 1:
        return root[0]
    return "(fp32 " + " ".join(root) + ")"
