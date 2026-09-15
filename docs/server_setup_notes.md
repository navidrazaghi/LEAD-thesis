# Running on the lab A100

Environment facts found on `razzaghi@213.233.184.253`, and what each one costs
if it is missed. Nothing here is a LEAD bug; it is all local machine state.

## Environment variables every run needs

```bash
export TIMM_USE_OLD_CACHE=1                       # weights from ~/.cache/torch, not the Hub
export LIBRARY_PATH=$HOME/.local/cuda-stubs:$LIBRARY_PATH   # lets Triton link
export LEAD_RUNTIME_TYPE_CHECKING=false           # only when torch.compile is on
```

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

## Disk

`/` is 1.1 T with **97 G free** (91% full); `/home/razzaghi` is already 351 G, of
which CARLA is 44 G. There is no scratch volume. The full dataset does not fit
even if it could be downloaded, and Fail2Drive wants a second 44 G CARLA build.
Plan on a subset from the start.

## CARLA renders on the CPU

The running server was started with `-RenderOffScreen` and picked the Mesa
software rasterizer instead of the A100. Evidence: `CarlaUE4-Linux-Shipping`
does not appear in `nvidia-smi` while the A100 sits at 0% utilization; the
process holds 400% CPU, 248 threads and 17 GB RSS; it has `libvulkan_lvp.so`
(lavapipe) mapped; and `~/.cache/mesa_shader_cache` is populated.

The cost is total: a run logged `System time = 5187 s` against
`Game time = 1.150 s`, a ratio of 0.000x. At that rate one Bench2Drive route
takes on the order of two weeks, so closed-loop evaluation cannot happen at all
until this is fixed.

The NVIDIA Vulkan ICD is installed and its library resolves, so the first thing
to try is forcing it and dropping every software driver from consideration:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

Then restart CARLA and check that `CarlaUE4` appears in `nvidia-smi`. Untested
so far: the currently running evaluation was left alone deliberately.
