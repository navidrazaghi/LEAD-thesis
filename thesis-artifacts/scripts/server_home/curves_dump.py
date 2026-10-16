import sys, pathlib, csv
sys.path.insert(0, str(pathlib.Path.home()))
import live_loss_plot as L
rows=[]
for label,(pre,post,_) in L.RUNS.items():
    if label.startswith("old"): continue
    for stage,run in (("pretrain",pre),("posttrain",post)):
        c=L.curve(run)
        for e in sorted(c): rows.append((label,stage,e,c[e][0]))
        if c:
            es=sorted(c); v=[c[e][0] for e in es]
            print(f"{label:28s} {stage:9s} epochs={len(es)} first={v[0]:.4f} e10={c.get(10,[float('nan')])[0]:.4f} e20={c.get(20,[float('nan')])[0]:.4f} last={v[-1]:.4f} min={min(v):.4f}@{es[v.index(min(v))]}")
with open("/tmp/curves_4models.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["model","stage","epoch","train_objective"]); w.writerows(rows)
L.draw()
