# Total: 6963.40 / 10000.00
# ID: sorted_chunks32_fp32_flat
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
    return "(fp32 " + " ".join(chunks) + ")"
