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

## CARLA renders on the CPU: the account has no access to the render node

**Fix**, as root, then log out and back in so the groups take effect:

```bash
sudo usermod -aG video,render razzaghi
```

### What happens without it

CARLA falls back to the Mesa software rasterizer. `CarlaUE4-Linux-Shipping` does
not appear in `nvidia-smi` while the A100 sits at 0% utilization; the process
holds 400% CPU, 248 threads and 17 GB RSS; `libvulkan_lvp.so` (lavapipe) is
mapped; `~/.cache/mesa_shader_cache` fills up. One run logged
`System time = 5187 s` against `Game time = 1.150 s` — a ratio of 0.000x, which
puts a single Bench2Drive route somewhere around two weeks.

### Why it happens

CUDA and graphics reach the GPU through different device nodes, and only one of
them is open to this account:

| node | access | used by |
| :--- | :----- | :------ |
| `/dev/nvidia0` | `crw-rw-rw-`, world | CUDA — which is why training works |
| `/dev/dri/card1` | ACL: root and gdm only | Vulkan |
| `/dev/dri/renderD129` | ACL: root and gdm only | Vulkan |

So the NVIDIA Vulkan driver cannot open its device and refuses to initialize,
Vulkan enumerates no NVIDIA GPU, and the loader hands UE4 the one driver that
needs no device node at all: lavapipe, on the CPU.

Forcing the ICD does **not** help — that was the first guess and it is wrong:

```
$ VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json vkprobe
vkCreateInstance failed: -9        # VK_ERROR_INCOMPATIBLE_DRIVER
```

Nor is it a version or packaging problem. `libGLX_nvidia.so.0` exports
`vk_icdGetInstanceProcAddr` and `vk_icdNegotiateLoaderICDInterfaceVersion`, has
no missing dependencies, and kernel module and userspace both read 595.71.05
from the same `nvidia-driver-595-server` package. Calling the negotiation entry
point directly returns `VK_ERROR_INITIALIZATION_FAILED` (-3) for every interface
version from 1 to 7, and every `vk_icdGetInstanceProcAddr` lookup returns NULL.
A driver that cannot open its device fails exactly this way.

The host is a VMware VM (`/dev/dri/card0` belongs to `vmwgfx`) with the A100
passed through on `card1`, which is the kind of setup where the render node ends
up owned by the display manager and nobody else.

### Confirming the fix

```bash
$ groups                     # must now list video and render
$ vkprobe                    # must list the A100 as DISCRETE_GPU, not llvmpipe
```

`vkprobe` is a ~30-line Vulkan enumeration probe; rebuild it any time with
`gcc probe.c -o vkprobe /usr/lib/x86_64-linux-gnu/libvulkan.so.1`. Before the
fix it reports one device, `llvmpipe (LLVM 20.1.2, 256 bits)`, type `CPU`.

Then restart CARLA and check that `CarlaUE4` appears in `nvidia-smi`.
