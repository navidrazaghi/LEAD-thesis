"""Recompute every closed-loop comparison three ways, and correct for the family.

The thesis reports paired t intervals. Three questions were raised about them
and each is answered here rather than asserted: whether the conclusions survive
a test that does not assume normality, whether they survive a correction for
having looked six times, and what the exact p-values are.
"""

import csv
import math
import pathlib
import random
import statistics
import sys

random.seed(20260820)
ROUNDS = 20000
# Relative to the repository, not to one machine's home directory: an absolute
# path here published the account name it was written on.
DEFAULT = pathlib.Path(__file__).resolve().parents[2] / "results" / "closed_loop.csv"
PATH = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT

with PATH.open(encoding="utf-8") as handle:
    rows = [
        r for r in csv.DictReader(handle) if (r.get("driving_score") or "").strip()
    ]
by = {}
for r in rows:
    by[(r["model"], r["modality"] + ":" + r["severity"], r["route"])] = float(
        r["driving_score"]
    )
models = ["rung0", "rung2a", "rung4"]
conds = ["none:0", "lidar:1.0", "camera:1.0"]


def t_cdf(t, df):
    """Two-sided p from a t statistic, via the incomplete beta."""
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    # continued fraction for the regularised incomplete beta
    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            delt = d * c
            h *= delt
            if abs(delt - 1.0) < EPS:
                break
        return h

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    return front * betacf(a, b, x)


def wilcoxon_p(d):
    """Two-sided signed-rank p, normal approximation with tie correction."""
    nz = [v for v in d if v != 0]
    n = len(nz)
    if n < 6:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    wp = sum(r for v, r in zip(nz, ranks, strict=True) if v > 0)
    mean = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (wp - mean) / sd
    return math.erfc(abs(z) / math.sqrt(2))


results = []
for cond in conds:
    routes = sorted({k[2] for k in by if k[1] == cond})
    common = [r for r in routes if all((m, cond, r) in by for m in models)]
    for m in ("rung0", "rung4"):
        d = [by[(m, cond, r)] - by[("rung2a", cond, r)] for r in common]
        n = len(d)
        mean = statistics.mean(d)
        sd = statistics.stdev(d)
        se = sd / math.sqrt(n)
        t = mean / se
        p = t_cdf(abs(t), n - 1)
        boot = []
        for _ in range(ROUNDS):
            s = [d[random.randrange(n)] for _ in range(n)]
            boot.append(sum(s) / n)
        boot.sort()
        lo, hi = boot[int(0.025 * ROUNDS)], boot[int(0.975 * ROUNDS)]
        results.append(
            {
                "comp": f"{m} vs rung2a, {cond}",
                "n": n,
                "mean": mean,
                "t": t,
                "p": p,
                "wp": wilcoxon_p(d),
                "blo": lo,
                "bhi": hi,
            }
        )

print(
    "  comparison                          n    mean      p(t)   p(wilcoxon)  bootstrap 95%"
)
for r in results:
    print(
        f"  {r['comp']:34} {r['n']:2} {r['mean']:+7.2f}  {r['p']:.4f}   {r['wp']:.4f}      "
        f"[{r['blo']:+6.2f},{r['bhi']:+6.2f}]"
    )

print()
print("  HOLM step-down over the family of six")
order = sorted(results, key=lambda r: r["p"])
k = len(order)
prev = 0.0
for i, r in enumerate(order):
    adj = min(1.0, max(prev, (k - i) * r["p"]))
    prev = adj
    print(
        f"    {r['comp']:34} raw p={r['p']:.4f}  Holm p={adj:.4f}  "
        f"{'survives 0.05' if adj < 0.05 else 'does not survive'}"
    )
