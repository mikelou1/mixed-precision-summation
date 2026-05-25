import math
import struct
from typing import List

_F = struct.Struct('f')
_fp = _F.pack
_fu = _F.unpack
_H = struct.Struct('<e')
_hp = _H.pack
_hu = _H.unpack

def _f(x):
    return _fu(_fp(x))[0]

def _h(x):
    return _hu(_hp(_f(x)))[0]

def _f32_pair(vals):
    vals = list(vals)
    while len(vals) > 1:
        m = len(vals) & ~1
        nxt = []
        app = nxt.append
        i = 0
        while i < m:
            app(_f(vals[i] + vals[i + 1]))
            i += 2
        if m < len(vals):
            app(vals[-1])
        vals = nxt
    return vals[0] if vals else 0.0

def _score_from_out(out, total, n, groups):
    if groups <= 0:
        return -1.0
    C = (n - groups) + 2 * (groups - 1)
    eta = abs(out - total) / (total + 1e-10)
    if eta <= 0.0:
        a = 1.0
    else:
        a = -math.log2(eta) / 24.0
        if a < 0.0:
            a = 0.0
        elif a > 1.0:
            a = 1.0
    return a * ((n - 1) / C)

def _eval_nodes(levels, nodes, total, n, has_zero=False):
    vals = [0.0] if has_zero else []
    vals.extend(levels[l][i] for l, i in nodes)
    out = _f32_pair(vals)
    return _score_from_out(out, total, n, len(vals)), out

def _half_list(values):
    L = len(values)
    if L == 0:
        return []
    try:
        fmtf = struct.Struct('<%df' % L)
        fmth = struct.Struct('<%de' % L)
        fv = fmtf.unpack(fmtf.pack(*values))
        return list(fmth.unpack(fmth.pack(*fv)))
    except OverflowError:
        out = []
        app = out.append
        for x in values:
            try:
                app(_h(x))
            except OverflowError:
                app(math.inf)
        return out

def _build_levels(hv, max_b):
    levels = [hv]
    cur = hv
    for _ in range(max_b):
        L = len(cur)
        m = L & ~1
        if m:
            sums = [cur[i] + cur[i + 1] for i in range(0, m, 2)]
            nxt = _half_list(sums)
        else:
            nxt = []
        if m < L:
            nxt.append(cur[-1])
        levels.append(nxt)
        cur = nxt
    return levels

def _canonical(levels, l, i):
    while l > 0:
        ci = i + i
        if ci + 1 < len(levels[l - 1]):
            break
        i = ci
        l -= 1
    return (l, i)

def _node_start(nd):
    return nd[1] << nd[0]

def _coverage_ok(m, nodes):
    cur = 0
    for l, i in sorted(nodes, key=_node_start):
        st = i << l
        ed = st + (1 << l)
        if ed > m:
            ed = m
        if st != cur or ed <= st:
            return False
        cur = ed
    return cur == m

def _path_options(levels, node, want, target, max_depth, keep):
    vals = []
    stack = [(node[0], node[1], 0, 0.0, 0)]
    while stack:
        l, i, dep, acc, bits = stack.pop()
        if dep >= max_depth or l <= 0:
            continue
        ci = i + i
        low = levels[l - 1]
        if ci + 1 >= len(low):
            continue
        d = low[ci] + low[ci + 1] - levels[l][i]
        if not math.isfinite(d):
            continue
        nd = acc + d
        nb0 = bits << 1
        nb1 = nb0 | 1
        dep2 = dep + 1
        if nd * want > 0.0 or abs(nd) < 0.125:
            vals.append((nd, dep2, nb0))
        stack.append((l - 1, ci, dep2, nd, nb0))
        stack.append((l - 1, ci + 1, dep2, nd, nb1))

    out = [(0.0, 0, 0)]
    if not vals:
        return out
    seen = set()
    lim = max(20, keep)
    views = (
        lambda x: (-abs(x[0]) / x[1], x[1]),
        lambda x: (abs(target - abs(x[0])), x[1]),
        lambda x: (abs(x[0]), x[1]),
        lambda x: (x[1], abs(target - abs(x[0]))),
    )
    for key in views:
        vals.sort(key=key)
        for v, c, bits in vals[:lim]:
            k = (round(v, 6), c)
            if k not in seen:
                seen.add(k)
                out.append((v, c, bits))
                if len(out) >= keep:
                    return out
    return out

def _nodes_for_option(levels, node, cost, bits):
    l, i = node
    if cost <= 0:
        return [node]
    out = []
    for sh in range(cost - 1, -1, -1):
        bit = (bits >> sh) & 1
        left = i + i
        other = _canonical(levels, l - 1, left + (1 - bit))
        out.append(other)
        i = left + bit
        l -= 1
        l, i = _canonical(levels, l, i)
    out.append(_canonical(levels, l, i))
    return out

def _approx_score(err, total, n, groups):
    C = (n - groups) + 2 * (groups - 1)
    eta = abs(err) / (total + 1e-10)
    if eta <= 0.0:
        a = 1.0
    else:
        a = -math.log2(eta) / 24.0
        if a < 0.0:
            a = 0.0
        elif a > 1.0:
            a = 1.0
    return a * ((n - 1) / C)

def _prune(nxt, E, total, n, base_groups, cap):
    if len(nxt) <= cap:
        nxt.sort(key=lambda st: -_approx_score(E + st[0], total, n, base_groups + st[1]))
        return nxt
    pr = []
    seen = set()
    def add(st):
        k = (round((E + st[0]) * 4096.0), st[1])
        if k not in seen:
            seen.add(k)
            pr.append(st)
            return True
        return False
    nxt.sort(key=lambda st: -_approx_score(E + st[0], total, n, base_groups + st[1]))
    for st in nxt[:cap // 2]:
        add(st)
    nxt.sort(key=lambda st: (abs(E + st[0]), st[1]))
    for st in nxt[:cap // 3]:
        add(st)
    nxt.sort(key=lambda st: (st[1], abs(E + st[0])))
    for st in nxt[:cap // 3]:
        add(st)
    return pr[:cap]

def _frontier_search(n, total, order, hvals, has_zero=False, sparse=False):
    m = len(order)
    if m <= 1:
        return None
    max_b = min(16, max(1, (m - 1).bit_length()))
    levels = _build_levels(hvals, max_b)
    best = None
    if sparse:
        cand = (max_b, max_b - 1, max_b - 2, max_b - 3, 16, 15, 14, 13)
        bvals = tuple(dict.fromkeys(b for b in cand if b >= 1))
    else:
        bvals = (16, 15, 14)
    for b in bvals:
        if b >= len(levels):
            continue
        raw = levels[b]
        if not raw:
            continue
        nodes = []
        seen = set()
        bad = False
        for i, v in enumerate(raw):
            if not math.isfinite(v):
                bad = True
                break
            nd = _canonical(levels, b, i)
            if nd not in seen:
                seen.add(nd)
                nodes.append(nd)
        if bad:
            continue
        nodes.sort(key=_node_start)
        G0 = len(nodes) + (1 if has_zero else 0)
        if G0 <= 1 or G0 > 34:
            continue
        base_sc, base_out = _eval_nodes(levels, nodes, total, n, has_zero)
        local_sc = base_sc
        local_nodes = nodes
        E = base_out - total
        if E != 0.0 and math.isfinite(E):
            want = 1.0 if E < 0.0 else -1.0
            if b == 16:
                group_cap = 28 if not sparse else 30
                keep = 120
                beam_cap = 420
                exact_cap = 90
            elif b == 15:
                group_cap = 28 if not sparse else 30
                keep = 115
                beam_cap = 380
                exact_cap = 80
            elif b == 14:
                group_cap = 30 if not sparse else 31
                keep = 90
                beam_cap = 280
                exact_cap = 70
            else:
                group_cap = 32
                keep = 80
                beam_cap = 300
                exact_cap = 100
            max_extra = max(0, group_cap - G0)
            if max_extra > 0:
                per = abs(E) / max(1, len(nodes))
                opts = [_path_options(levels, nd, want, per, min(13, nd[0]), keep) for nd in nodes]
                beam = [(0.0, 0, ())]
                for bi, olist in enumerate(opts):
                    nxt = []
                    for d0, c0, ch0 in beam:
                        for oi, (dv, dc, bits) in enumerate(olist):
                            nc = c0 + dc
                            if nc <= max_extra:
                                if dc:
                                    nxt.append((d0 + dv, nc, ch0 + ((bi, oi),)))
                                else:
                                    nxt.append((d0, c0, ch0))
                    if not nxt:
                        break
                    beam = _prune(nxt, E, total, n, G0, beam_cap)
                for _, _, ch in beam[:exact_cap]:
                    cmap = {bi: opts[bi][oi] for bi, oi in ch}
                    ns = []
                    for bi, nd in enumerate(nodes):
                        if bi in cmap:
                            _, dc, bits = cmap[bi]
                            ns.extend(_nodes_for_option(levels, nd, dc, bits))
                        else:
                            ns.append(nd)
                    ns.sort(key=_node_start)
                    sc2, _ = _eval_nodes(levels, ns, total, n, has_zero)
                    if sc2 > local_sc:
                        local_sc = sc2
                        local_nodes = ns
        if not _coverage_ok(m, local_nodes):
            continue
        if best is None or local_sc > best[0]:
            best = (local_sc, order, levels, local_nodes, has_zero)
    return best

def _make_pair_tree(items, prec):
    items = list(items)
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    pref = '(' + prec + ' '
    while len(items) > 1:
        L = len(items)
        nxt = []
        app = nxt.append
        i = 0
        while i + 1 < L:
            app(pref + items[i] + ' ' + items[i + 1] + ')')
            i += 2
        if i < L:
            app(items[i])
        items = nxt
    return items[0]

def _emit_power(tokens, order, st, size):
    if size == 1:
        tokens.append(str(order[st] + 1))
        return
    h = size >> 1
    tokens.append('(fp16 ')
    _emit_power(tokens, order, st, h)
    tokens.append(' ')
    _emit_power(tokens, order, st + h, h)
    tokens.append(')')

def _emit_node(tokens, order, node):
    l, i = node
    st = i << l
    ed = min(len(order), st + (1 << l))
    if ed <= st:
        return
    k = ed - st
    if k == 1:
        tokens.append(str(order[st] + 1))
    elif k == (1 << l):
        _emit_power(tokens, order, st, k)
    else:
        tokens.append(_make_pair_tree((str(order[j] + 1) for j in range(st, ed)), 'fp16'))

def _build_answer(plan, zeros):
    sc, order, levels, nodes, has_zero = plan
    pieces = []
    if zeros:
        if len(zeros) == 1:
            pieces.append(str(zeros[0] + 1))
        else:
            pieces.append('(fp16 ' + ' '.join(str(i + 1) for i in zeros) + ')')
    for nd in nodes:
        toks = []
        _emit_node(toks, order, nd)
        if toks:
            pieces.append(''.join(toks))
    return _make_pair_tree(pieces, 'fp32')

def _choose(items, T, SCALE, MARGIN):
    items.sort(reverse=True)

    R = int((T + MARGIN) * SCALE + 1.999999)
    if R < 1:
        R = 1

    TU = int(T * SCALE + 0.5)
    if TU > R:
        TU = R

    mask = (1 << (R + 1)) - 1
    bits = 1
    pref = [bits]

    for u, _ in items:
        if u <= R:
            bits |= (bits << u) & mask
        pref.append(bits)

    best = TU
    if not ((bits >> best) & 1):
        lo = best - 1
        hi = best + 1
        while lo >= 0 or hi <= R:
            if lo >= 0 and ((bits >> lo) & 1):
                best = lo
                break
            if hi <= R and ((bits >> hi) & 1):
                best = hi
                break
            lo -= 1
            hi += 1

    selected = []
    s = best
    hi = len(items)

    while s > 0 and hi > 0:
        lo = 1
        r = hi
        while lo < r:
            mid = (lo + r) // 2
            if (pref[mid] >> s) & 1:
                r = mid
            else:
                lo = mid + 1

        j = lo
        if j > hi or not ((pref[j] >> s) & 1):
            break

        i = j - 1
        u, bi = items[i]
        selected.append(bi)
        s -= u
        hi = i

    return selected

def _fallback_solve(n: int, values: List[float]) -> str:
    if n < 1000:
        return "(fp32 " + " ".join(str(i) for i in range(1, n + 1)) + ")"

    B = 1280
    L = 192
    SCALE = 64.0
    MARGIN = 2000.0

    vals = values
    order = list(range(n))
    order.sort(key=vals.__getitem__)
    total = math.fsum(vals)

    hv0 = _half_list(vals)
    hvals = [hv0[i] for i in order]

    starts = list(range(0, n, B))
    g = len(starts)
    full = n // B
    rem = n - full * B

    acc = tuple(hvals[0::B])
    left2_full = None
    left2_last = None
    mid_full = B // 2
    mid_last = rem // 2 if rem else 0

    for off in range(1, B):
        vo = hvals[off::B]
        k = len(vo)

        if k == g:
            acc = tuple(_half_list([acc[i] + vo[i] for i in range(g)]))
        elif k:
            pre = tuple(_half_list([acc[i] + vo[i] for i in range(k)]))
            acc = pre + acc[k:]

        if off == mid_full - 1:
            left2_full = acc[:full]
        if rem and off == mid_last - 1:
            left2_last = acc[full]

    allv = list(acc)

    if left2_full is None:
        left2_full = tuple(hvals[0:full * B:B])
    if rem and left2_last is None:
        left2_last = hvals[full * B]

    def seg_full(a, b):
        if not full:
            return ()
        cur = tuple(hvals[a:full * B:B])
        for off in range(a + 1, b):
            vo = hvals[off:full * B:B]
            cur = tuple(_half_list([cur[i] + vo[i] for i in range(full)]))
        return cur

    right2 = seg_full(mid_full, B) if full else ()

    partials = []
    for i in range(0, g, L):
        partials.append(_f32_pair(allv[i:i + L]))

    base = sum(partials)
    target = total - base
    sign = 1.0 if target >= 0.0 else -1.0
    T = abs(target)

    p2_delta = [0.0] * g
    items = []

    for bi in range(full):
        d = left2_full[bi] + right2[bi] - allv[bi]
        p2_delta[bi] = d
        x = d * sign
        if x > 0.0:
            u = int(x * SCALE + 0.5)
            if u > 0:
                items.append((u, bi))

    if rem:
        bi = full
        st = full * B
        k = rem
        mid = st + k // 2
        ed = n

        r = hvals[mid]
        for p in range(mid + 1, ed):
            r = _h(r + hvals[p])

        d = left2_last + r - allv[bi]
        p2_delta[bi] = d
        x = d * sign
        if x > 0.0:
            u = int(x * SCALE + 0.5)
            if u > 0:
                items.append((u, bi))

    sel2 = set(_choose(items, T, SCALE, MARGIN))

    def eval_choice(pcmap, segvals=None):
        outs = []

        for bi in range(g):
            pc = pcmap.get(bi, 1)

            if pc == 2:
                if bi < full:
                    outs.append(left2_full[bi])
                    outs.append(right2[bi])
                else:
                    st = full * B
                    k = rem
                    mid = st + k // 2
                    ed = n

                    la = left2_last
                    r = hvals[mid]
                    for p in range(mid + 1, ed):
                        r = _h(r + hvals[p])

                    outs.append(la)
                    outs.append(r)

            elif pc > 2 and segvals is not None:
                outs.extend(segvals[bi])
            else:
                outs.append(allv[bi])

        ps = []
        for i in range(0, len(outs), L):
            ps.append(_f32_pair(outs[i:i + L]))

        S = sum(ps)
        m = len(ps)
        groups = len(outs)
        C = (n - groups) + 2 * (groups - m) + 8 * (m - 1)

        eta = abs(S - total) / (total + 1e-10)
        alpha = 1.0 if eta == 0.0 else min(1.0, max(0.0, -math.log2(eta) / 24.0))
        beta = (n - 1) / C

        return alpha * beta, eta

    pc2 = {bi: 2 for bi in sel2}
    best_pc = pc2
    best_score, best_eta = eval_choice(pc2)

    if best_eta > 6.2e-8:
        bestx = p2_delta[:]
        bestpc = [2 if d != 0.0 else 1 for d in p2_delta]

        for pc in (4, 8):
            if full:
                seglen = B // pc
                segs = []

                for q in range(pc):
                    segs.append(seg_full(q * seglen, (q + 1) * seglen))

                for i in range(full):
                    ss = 0.0
                    for q in range(pc):
                        ss += segs[q][i]

                    d = ss - allv[i]
                    if abs(d) > abs(bestx[i]):
                        bestx[i] = d
                        bestpc[i] = pc

            if rem and pc <= rem:
                bi = full
                st = full * B
                k = rem
                ss = 0.0

                for q in range(pc):
                    a = st + (q * k) // pc
                    b = st + ((q + 1) * k) // pc
                    cur = hvals[a]

                    for p in range(a + 1, b):
                        cur = _h(cur + hvals[p])

                    ss += cur

                d = ss - allv[bi]
                if abs(d) > abs(bestx[bi]):
                    bestx[bi] = d
                    bestpc[bi] = pc

        items = []
        for bi, d in enumerate(bestx):
            x = d * sign
            if x > 0.0:
                u = int(x * SCALE + 0.5)
                if u > 0:
                    items.append((u, bi))

        sel = set(_choose(items, T, SCALE, MARGIN))
        pcm = {bi: bestpc[bi] for bi in sel}

        segvals = {}
        for bi, pc in pcm.items():
            if pc <= 2:
                continue

            st = bi * B
            ed = min(n, st + B)
            k = ed - st
            arr = []

            for q in range(pc):
                a = st + (q * k) // pc
                b = st + ((q + 1) * k) // pc
                cur = hvals[a]

                for p in range(a + 1, b):
                    cur = _h(cur + hvals[p])

                arr.append(cur)

            segvals[bi] = arr

        sc, _ = eval_choice(pcm, segvals)
        if sc > best_score:
            best_pc = pcm

    so = [str(i + 1) for i in order]
    groups = []

    for bi, st in enumerate(starts):
        ed = min(n, st + B)
        k = ed - st
        pc = best_pc.get(bi, 1)

        if pc <= 1 or pc > k:
            groups.append("(fp16 " + " ".join(so[st:ed]) + ")")
        else:
            for q in range(pc):
                a = st + (q * k) // pc
                b = st + ((q + 1) * k) // pc
                groups.append("(fp16 " + " ".join(so[a:b]) + ")")

    partial = []
    for i in range(0, len(groups), L):
        part = groups[i:i + L]
        partial.append(part[0] if len(part) == 1 else _make_pair_tree(part, "fp32"))

    return partial[0] if len(partial) == 1 else "(fp64 " + " ".join(partial) + ")"

def solve(n: int, values: List[float]) -> str:
    total = math.fsum(values)
    order = list(range(n))
    order.sort(key=values.__getitem__)
    hv_unsorted = _half_list(values)

    dense_order = order
    mx = values[order[-1]] if order else 0.0
    if mx > 0.0:
        q1 = values[order[n >> 2]]
        q2 = values[order[n >> 1]]
        if q1 > 0.010 * mx and q2 > 0.080 * mx:
            rem16 = n & ((1 << 16) - 1)
            if rem16:
                st = (n - rem16) >> 1
                dense_order = order[:st] + order[st + rem16:] + order[st:st + rem16]

    hvals = [hv_unsorted[i] for i in dense_order]

    best = _frontier_search(n, total, dense_order, hvals, False, False)
    best_zeros = []

    zc = 0
    for x in values:
        if x == 0.0:
            zc += 1
    if zc > n // 8 and zc < n - 1:
        zeros = []
        active = []
        for i, x in enumerate(values):
            if x == 0.0:
                zeros.append(i)
            else:
                active.append(i)
        active.sort(key=values.__getitem__)
        ah = [hv_unsorted[i] for i in active]
        sp = _frontier_search(n, total, active, ah, True, True)
        if sp is not None and (best is None or sp[0] > best[0]):
            best = sp
            best_zeros = zeros

    if best is not None and best[0] >= 0.9990:
        return _build_answer(best, best_zeros)
    return _fallback_solve(n, values)
