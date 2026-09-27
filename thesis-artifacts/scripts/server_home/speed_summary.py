"""Steady-state training throughput from a run's progress-bar output.

The bar counts samples and prints elapsed time, so the rate between two points
well past the start excludes compilation and autotuning; the elapsed time at the
first moving point measures that start-up cost on its own. GPU and CPU samples
taken during the run say which side was the limit.

Usage: speed_summary.py LABEL TRAIN_LOG GPU_LOG VMSTAT_LOG
"""
import re
import statistics
import sys

label, train_log, gpu_log, vm_log = sys.argv[1:5]
text = open(train_log, encoding="utf-8", errors="replace").read()
pattern = re.compile(r"Epoch 0:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[(?:(\d+):)?(\d+):(\d+)<")
points = []
for chunk in re.split(r"[\r\n]", text):
    match = pattern.search(chunk)
    if match:
        done, total = int(match[1]), int(match[2])
        seconds = int(match[3] or 0) * 3600 + int(match[4]) * 60 + int(match[5])
        points.append((done, total, seconds))
if not points:
    print(f"{label}: no progress found in {train_log}")
    sys.exit(0)
total = points[-1][1]
moving = [p for p in points if p[0] > 0]
startup = moving[0][2] if moving else float("nan")
steady = [p for p in points if p[0] >= 0.25 * total]
if len(steady) >= 2 and steady[-1][2] > steady[0][2]:
    rate = (steady[-1][0] - steady[0][0]) / (steady[-1][2] - steady[0][2])
else:
    rate = float("nan")

gpu_util, gpu_mem = [], []
for line in open(gpu_log):
    parts = [p.strip().split()[0] for p in line.split(",")]
    if len(parts) == 2 and parts[0].isdigit():
        gpu_util.append(int(parts[0]))
        gpu_mem.append(int(parts[1]))
busy = [u for u in gpu_util if u > 0]
idle = []
for line in open(vm_log):
    fields = line.split()
    if len(fields) >= 15 and fields[14].isdigit():
        idle.append(int(fields[14]))

print(f"{label:28s} steady {rate:6.1f} samples/s | first step after {startup:4.0f} s | "
      f"GPU util median {statistics.median(busy) if busy else 0:3.0f}% | "
      f"GPU mem peak {max(gpu_mem) / 1024 if gpu_mem else 0:4.1f} GB | "
      f"CPU idle median {statistics.median(idle) if idle else float('nan'):3.0f}%")
