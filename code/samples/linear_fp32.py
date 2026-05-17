# Total: 3692.49 / 10000.00
# ID: linear_fp32
def solve(n, values):
    if n == 1:
        return "1"
    return "(fp32 " + " ".join(str(i) for i in range(1,n+1)) + ")"
