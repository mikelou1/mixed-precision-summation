# Total: 6964.64 / 10000.00
# ID: sorted_chunks32_fp32_hierarchical_fanout96
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
    fan = 96
    while len(chunks) > fan:
        nxt = []
        for i in range(0,len(chunks),fan):
            part = chunks[i:i+fan]
            if len(part) > 1:
                nxt.append("(fp32 " + " ".join(part) + ")")
            else:
                nxt.append(part[0])
        chunks = nxt
    if len(chunks) == 1:
        return chunks[0]
    return "(fp32 " + " ".join(chunks) + ")"
