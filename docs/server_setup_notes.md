# Running on the lab A100

Environment facts found on `razzaghi@213.233.184.253`, and what each one costs
if it is missed. Nothing here is a LEAD bug; it is all local machine state.

## Launching a training run

Use `~/run_rung.sh <output-name> [overrides...]`. It carries the environment
below, so no run can accidentally start without it. Everything here was found
the hard way; each line is load-bearing.

```bash
ulimit -n 65536                                   # see below — the expensive one
export TIMM_USE_OLD_CACHE=1                       # weights from ~/.cache/torch, not the Hub
export LEAD_RUNTIME_TYPE_CHECKING=false           # only when torch.compile is on
export LIBRARY_PATH=$HOME/.local/cuda-stubs:$LIBRARY_PATH   # redundant since the driver reinstall
```

**`ulimit -n 65536` is not optional, and skipping it does not fail loudly.**
The cache store opens an LMDB environment per log, and with 450 logs across the
dataloader workers the default soft limit of 1024 runs out. Training then emits
`OSError: [Errno 24] Too many open files`, restarts workers, and *keeps going*
— at 0.06 steps/s instead of 3.15, a 50x slowdown that looks like ordinary GPU
contention. Measured on this machine:

| | steps/s | one epoch |
| :--- | ---: | ---: |
| default soft limit 1024 | 0.06 | 34 h |
| raised to 65536 | 3.15–5.9 | 25–40 min |

The hard limit is 1048576, so raising the soft limit needs no root.

**`TIMM_USE_OLD_CACHE=1`** — the backbone builds `resnet34` with
`pretrained=True`, which reaches for huggingface.co. That host is unreachable
from this machine. The weights are already in `~/.cache/torch/hub/checkpoints/`,
and this flag is what makes timm read them from there. Without it, anything that
constructs a policy dies at import of the backbone, including seven config
tests that look unrelated.

**`LIBRARY_PATH`** — `torch.compile` builds Triton kernels with
`gcc ... -lcuda`, and the linker needs an unversioned `libcuda.so`. The driver
ships `libcuda.so.1` only, so the link fails and compilation falls over.
The symlink is already created:

```bash
mkdir -p ~/.local/cuda-stubs
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 ~/.local/cuda-stubs/libcuda.so
```

Training defaults to `use_torch_compile=true`, so without this the default
training command fails outright.

**`LEAD_RUNTIME_TYPE_CHECKING=false`** — beartype and Dynamo cannot run
together. `scripts/common/pretrain.sh` already clears it; anything else that
compiles has to clear it too. Leave it on for tests, where it is doing useful
work.

## Network

| host | reachable | consequence |
| :--- | :-------- | :---------- |
| github.com | yes | |
| pypi.org | no | use the Tsinghua mirror, as `~/.pip/pip.conf` already does |
| huggingface.co | no | **the dataset cannot be downloaded here** |
| hf-mirror.com | root only, CDN 403 | does not help for model or dataset files |

The 1.1 TB Py123D dataset is HuggingFace-hosted, and neither the direct host nor
the mirror's file CDN is reachable. Getting the data onto this machine needs a
route that does not exist yet — a proxy, or a transfer from somewhere that can
reach the Hub.

## The dataset subset, and what it cost to make usable

`scripts/common/fetch_dataset_subset.py` pulls a stratified 450-log subset
(18 GB) rather than the full release, which does not fit. Four things bit on
the way from "downloaded" to "training runs", none of which a unit test would
have caught:

1. **38 of the released logs ship without `sync.arrow`.** The scene index finds
   nothing in them and the cache build dies with `no scenes match splits`,
   which reads like the data root is wrong. The fetcher now skips them.
2. **The depth cameras are not optional in practice.** Dropping them saves 40%
   of the download, and the cache build still succeeds — because `depth_target`
   is not a cacheable part, so it never runs there. Training reads it live and
   dies on the first batch. Fetch with `--keep-depth` unless you also set
   `policy.transfuser.use_depth=false`.
3. **The cache must be rebuilt when the log set changes.** Swapping six broken
   logs for six good ones left the cache covering a set that no longer matched,
   and training failed on a missing LMDB directory.
4. **The file-descriptor limit**, above.

The lesson behind all four: a green cache build says nothing about whether
training will run. It exercises a strict subset of the data path.

## Disk

`/` is 1.1 T with **97 G free** (91% full); `/home/razzaghi` is already 351 G, of
which CARLA is 44 G. There is no scratch volume. The full dataset does not fit
even if it could be downloaded, and Fail2Drive wants a second 44 G CARLA build.
Plan on a subset from the start.

## CARLA rendered on the CPU — fixed

Two separate faults, both now repaired. CARLA runs on the A100.

**Fault 1 — the account could not open the render node.**

```bash
sudo usermod -aG video,render razzaghi     # then log out and back in
```

`/dev/dri/card1` and `/dev/dri/renderD129` opened afterwards, but Vulkan still
enumerated only llvmpipe: necessary, not sufficient.

**Fault 2 — the driver install was damaged.** `dpkg -V` reported seven files the
packages had installed and that were no longer on disk:

| package | missing |
| :------ | :------ |
| `libnvidia-compute-595-server` | `libcuda.so`, `libnvidia-ml.so`, `libnvidia-nvvm.so`, `libnvidia-ptxjitcompiler.so` |
| `libnvidia-gl-595-server` | `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`, `/usr/share/egl/egl_external_platform.d/15_nvidia_gbm.json`, `nvidia/wine/nvngx.dll` |

Every one is an unversioned development symlink or a GLVND/EGL vendor
registration — the shape of a cleanup that stripped the graphics stack and took
the `.so` dev links with it. The missing `libcuda.so` is the same fault that
breaks Triton, so the `~/.local/cuda-stubs` workaround above is patching a
symptom of this.

Restoring them fixed it:

```bash
sudo apt install --reinstall libnvidia-gl-595-server libnvidia-compute-595-server
sudo ldconfig
```

The two EGL registrations were the cause. Without them the NVIDIA driver had no
registered rendering path, so its Vulkan ICD refused to initialize and the
loader fell through to the one driver that needs no GPU.

`libcuda.so` came back in the same reinstall, so the `~/.local/cuda-stubs`
symlink above is now redundant. Harmless to keep; unnecessary to set.

### Confirmed working

```
$ vkprobe
physical devices: 2
  [0] type=DISCRETE_GPU    name=NVIDIA A100-SXM4-40GB
  [1] type=CPU             name=llvmpipe (LLVM 20.1.2, 256 bits)
```

The A100 enumerates first, which is what UE4 takes. A CARLA server started
afterwards showed up in `nvidia-smi` as `C+G` holding 6222 MiB, mapped
`libEGL_nvidia` / `libGLX_nvidia` / `libnvidia-egl-gbm` and no lavapipe, and sat
at 144% CPU instead of 400%.

### What it looked like while broken

CARLA falls back to the Mesa software rasterizer. `CarlaUE4-Linux-Shipping` does
not appear in `nvidia-smi` while the A100 sits at 0% utilization; the process
holds 400% CPU, 248 threads and 17 GB RSS; `libvulkan_lvp.so` (lavapipe) is
mapped; `~/.cache/mesa_shader_cache` fills up. One run logged
`System time = 5187 s` against `Game time = 1.150 s` — a ratio of 0.000x, which
puts a single Bench2Drive route somewhere around two weeks.

### What has been ruled out

Vulkan enumerates exactly one device, `llvmpipe (LLVM 20.1.2, 256 bits)`, type
`CPU`. The A100 is invisible to it, so the loader hands UE4 the one driver that
needs no GPU at all.

Forcing the ICD does not help, which was the first guess and was wrong:

```
$ VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json vkprobe
vkCreateInstance failed: -9        # VK_ERROR_INCOMPATIBLE_DRIVER
```

It is not a loader/driver version mismatch either. Calling
`vk_icdNegotiateLoaderICDInterfaceVersion` on `libGLX_nvidia.so.0` directly
returns `VK_ERROR_INITIALIZATION_FAILED` (-3) for every interface version from 1
to 7, and every `vk_icdGetInstanceProcAddr` lookup returns NULL. The library
exports all the right symbols and has no missing dependencies; kernel module and
userspace both read 595.71.05. The driver simply declines to start.

Nor is the GPU itself restricted: virtualization mode is `None`, MIG disabled,
compute mode `Default`. CUDA works, which is why training runs fine — CUDA goes
through `/dev/nvidia0` and never touches the graphics path.

The host is a VMware VM: `/dev/dri/card0` belongs to `vmwgfx` and the A100 is
passed through on `card1`.

### Confirming the fix

```bash
$ groups                     # must now list video and render
$ vkprobe                    # must list the A100 as DISCRETE_GPU, not llvmpipe
```

`vkprobe` is a ~30-line Vulkan enumeration probe; rebuild it any time with
`gcc probe.c -o vkprobe /usr/lib/x86_64-linux-gnu/libvulkan.so.1`. Before the
fix it reports one device, `llvmpipe (LLVM 20.1.2, 256 bits)`, type `CPU`.

Then restart CARLA and check that `CarlaUE4` appears in `nvidia-smi`.

## Reading a closed-loop result status

`TickRuntime` looks like a simulator fault and is not one. `scenario_manager`
raises it at a hard cap:

```python
self.tick_count += 1
if self.tick_count > 4000:
    raise TickRuntimeError("RuntimeError, tick_count > 4000")
```

So it means the agent spent its whole step budget without reaching the goal —
stalling or crawling. It belongs with `Agent got blocked`, not with the
infrastructure failures, and **the leaderboard records a full score for those
routes** (route completion 11–93% in the pilot, driving scores 7–34).

Treating it as infrastructure cost real time and nearly produced a wrong result:

- Those rows were dropped from the analysis, and dropped *unevenly* — six from
  the clean condition, four from `camera:0.5`, none from `camera:1.0`. Removing
  the worst runs from the clean condition while keeping them elsewhere made a
  degraded model look better than an intact one (mean DS 48.2 vs 32.9). With
  every row counted the ordering behaves sanely again.
- The sweep retried them, burning ~20 minutes per route to reproduce the same
  number. A retry landing on the same score is the signature of a real agent
  failure; genuine sim flakiness does not reproduce that cleanly.

The wall-clock tell: nothing that completed ran past 750 s, and nothing that hit
the cap ran under 890 s. Burning all 4000 ticks simply takes the longest.

Only `NoResult` and `Agent timed out` remain in `_INFRASTRUCTURE_FAILURES`.

## Sweep logs are block buffered

`run_evaluation.py` calls `_unbuffer_stdout()` for a reason. Redirected stdout
buffers by block, so a multi-hour sweep writes **nothing** to its log until it
exits. A healthy run looks dead, and worse, any `grep` over the log answers
confidently about an empty file — which is how a check for simulator failures
came back clean while failures were in fact accumulating.
