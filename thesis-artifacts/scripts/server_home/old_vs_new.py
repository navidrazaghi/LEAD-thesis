"""The published checkpoint on the same routes, old server against new.

reference_closed_loop.csv holds ref0 -- resnet34_v1.5.0 seed0 -- driven on the
old server, 30 degradation routes under three conditions. This machine drove the
same checkpoint on the same routes; pairing them route by route is the direct
test of whether the harness, including its sensor-degradation path, behaves the
same here.

Also reports test A: whether the smoke checkpoint set up and moved.
"""
import csv
import pathlib
import statistics as st

R = pathlib.Path("/home/new_drive/razaghi/lead")


def rows(path, model):
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(open(path)):
        if r["model"] == model:
            out[(r["modality"], float(r["severity"] or 0), r["route"])] = r
    return out


print("=== test A: our own checkpoints ===")
for model in ("smoke", "smoke_post"):
    for key, r in sorted(rows(R / "results/system_test_smoke_load.csv", model).items()):
        print(f"  {model:10s} {key[2]:11s} status {r['status']:32s} DS {float(r['driving_score']):6.2f}  "
              f"distance {r.get('distance_m') or '?'} m  ticks {r.get('ticks') or '?'}")

print("\n=== test B: same checkpoint, old server vs new ===")
old = rows(R / "thesis-artifacts/results/reference_closed_loop.csv", "ref0")
new = rows(R / "results/closed_loop_reference_v150.csv", "reference_v150")
for cond in (("none", 0.0), ("lidar", 1.0), ("camera", 1.0)):
    candidates = sorted(k for k in new if k[:2] == cond and k in old)
    # A row with no driving score is a run that produced no result; it cannot
    # be paired, so it is named rather than silently dropped.
    unscored = [k for k in candidates
                if not old[k]["driving_score"] or not new[k]["driving_score"]]
    pairs = [k for k in candidates if k not in unscored]
    for k in unscored:
        print(f"{cond[0]}:{cond[1]}  unpaired {k[2]}: old '{old[k]['driving_score']}' ({old[k]['status']}), "
              f"new '{new[k]['driving_score']}' ({new[k]['status']})")
    if not pairs:
        print(f"{cond[0]}:{cond[1]}  no pairs yet")
        continue
    o = [float(old[k]["driving_score"]) for k in pairs]
    n = [float(new[k]["driving_score"]) for k in pairs]
    d = [b - a for a, b in zip(o, n)]
    se = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
    same = sum(abs(x) < 1e-9 for x in d)
    print(f"{cond[0]}:{cond[1]:<4}  n={len(pairs)}  old {st.mean(o):6.2f}  new {st.mean(n):6.2f}  "
          f"diff {st.mean(d):+6.2f}  SE {se:5.2f}  t {st.mean(d) / se:5.2f}  identical {same}/{len(d)}")
    for k, x in sorted(zip(pairs, d), key=lambda t: -abs(t[1]))[:4]:
        if abs(x) > 1e-9:
            print(f"      {k[2]:11s} old {float(old[k]['driving_score']):6.2f} ({old[k]['status']})"
                  f"  new {float(new[k]['driving_score']):6.2f} ({new[k]['status']})")
