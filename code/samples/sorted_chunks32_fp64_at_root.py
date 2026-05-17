# Total: 6964.25 / 10000.00
# ID: sorted_chunks32_fp64_at_root
def solve(n, values):
    if n == 1:
        return "1"
    order = sorted(range(n), key=lambda i: values[i])
    labels = [str(i+1) for i in order]
    K = 32
    chunks = []
    for i in range(0,n,K):
        ch = labels[i:i+K]
        if len(ch) > 1:
            chunks.append("(fp16 " + " ".join(ch) + ")")
        else:
            chunks.append(ch[0])
    if len(chunks) == 1:
        return chunks[0]
    def build_tree(items):
        while len(items) > 1:
            nxt = []
            for i in range(0,len(items)-1,2):
                nxt.append("(fp32 " + items[i] + " " + items[i+1] + ")")
            if len(items) % 2:
                nxt.append(items[-1])
            items = nxt
        return items[0]
    mid = len(chunks) // 2
    left = build_tree(chunks[:mid])
    right = build_tree(chunks[mid:])
    return "(fp64 " + left + " " + right + ")"
