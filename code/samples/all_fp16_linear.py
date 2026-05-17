# Total: 9.02 / 10000.00
# ID: all_fp16_linear
def solve(n, values):
    if n == 1:
        return "1"
    return "(fp16 " + " ".join(str(i) for i in range(1,n+1)) + ")"
