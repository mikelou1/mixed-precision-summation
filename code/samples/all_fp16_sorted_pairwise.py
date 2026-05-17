# Total: 0.00 / 10000.00
# ID: all_fp16_sorted_pairwise
def solve(n, values):
    if n == 1:
        return "1"
    order = sorted(range(n), key=lambda i: values[i])
    items = [str(i+1) for i in order]
    while len(items) > 1:
        nxt = []
        for i in range(0,len(items)-1,2):
            nxt.append("(fp16 " + items[i] + " " + items[i+1] + ")")
        if len(items) % 2:
            nxt.append(items[-1])
        items = nxt
    return items[0]
