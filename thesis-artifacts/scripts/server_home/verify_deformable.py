# -*- coding: utf-8 -*-
"""Independent audit of the deformable fusion operator, at the real token sizes.

Written against the operator's contract rather than its code: the reference
aggregation here is the Deformable DETR PyTorch core, transcribed separately, so
agreement is evidence and not a restatement. Checks, in order:

  1. output equals the reference core on the same inputs
  2. sampling weights are a distribution over (modality, point)
  3. at initialization every query reads the uniform fan, as DETR intends
  4. every parameter takes gradient
  5. bf16 autocast and torch.compile agree with eager fp32
  6. a modality bias moves the mass it is supposed to move
  7. the calibrated reference table is in range, finite, and geometrically sane
  8. cost against the dense operator it replaces, at 552 tokens

Run from the repo root.
"""
import time

import torch
import torch.nn.functional as F

from lead.config import load_lead_config
from lead.policy.transfuser.encoder import fusion_geometry
from lead.policy.transfuser.encoder.deformable_attention import (
    MultiScaleDeformableAttention,
    default_reference_points,
)
from lead.policy.transfuser.encoder.transfuser_backbone import SelfAttention

IMAGE_SHAPE, BEV_SHAPE = (12, 36), (10, 12)
SHAPES = (IMAGE_SHAPE, BEV_SHAPE)
N_IMG = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
TOKENS = N_IMG + BEV_SHAPE[0] * BEV_SHAPE[1]
EMBD, HEADS, POINTS = 64, 4, 4
ok = True


def check(name: str, passed: bool, detail: str = "") -> None:
    """Record and print one check."""
    global ok
    ok = ok and passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")


def detr_core(value, spatial_shapes, sampling_locations, attention_weights):
    """Deformable DETR's ms_deform_attn_core_pytorch, transcribed independently.

    Args:
        value: (B, T, heads, dim) per-head values, levels concatenated.
        spatial_shapes: per-level (height, width).
        sampling_locations: (B, T, heads, L, K, 2) in [0, 1], (x, y).
        attention_weights: (B, T, heads, L, K) normalized over (L, K).

    Returns:
        (B, T, heads*dim) aggregated features.
    """
    bs, _, n_heads, dim = value.shape
    _, n_query, _, n_levels, n_points, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    out = []
    for lid, (h, w) in enumerate(spatial_shapes):
        v = value_list[lid].flatten(2).transpose(1, 2).reshape(bs * n_heads, dim, h, w)
        g = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        out.append(
            F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False),
        )
    w_ = attention_weights.transpose(1, 2).reshape(
        bs * n_heads, 1, n_query, n_levels * n_points,
    )
    return (
        (torch.stack(out, dim=-2).flatten(-2) * w_)
        .sum(-1)
        .view(bs, n_heads * dim, n_query)
        .transpose(1, 2)
    )


torch.manual_seed(0)
attn = MultiScaleDeformableAttention(
    n_embd=EMBD, n_head=HEADS, attn_pdrop=0.0, resid_pdrop=0.0,
    spatial_shapes=SHAPES, num_points=POINTS,
).eval().double()
x = torch.randn(2, TOKENS, EMBD, dtype=torch.float64)

# --- 1, 2, 3: the operator's own arithmetic --------------------------------
with torch.no_grad():
    value = attn.value_proj(x).view(2, TOKENS, HEADS, EMBD // HEADS)
    offsets = attn.sampling_offsets(x).view(2, TOKENS, HEADS, 2, POINTS, 2)
    logits = attn.attention_weights(x).view(2, TOKENS, HEADS, 2, POINTS)
    weights = F.softmax(logits.flatten(-2), dim=-1).view(2, TOKENS, HEADS, 2, POINTS)
    reference = attn.reference_points(x)
    locations = (
        reference[:, :, None, :, None, :]
        + offsets / attn.offset_normalizer[None, None, None, :, None, :]
    )
    mine = attn.proj(
        __import__("lead.policy.transfuser.encoder.deformable_attention", fromlist=["x"])
        .deformable_aggregate(value, SHAPES, locations, weights),
    )
    theirs = attn.proj(detr_core(value, SHAPES, locations, weights))
    check("matches Deformable DETR reference core",
          torch.allclose(mine, theirs, atol=1e-10),
          f"max abs diff {float((mine - theirs).abs().max()):.2e}")
    check("weights are a distribution over (modality, point)",
          torch.allclose(weights.sum(dim=(-2, -1)), torch.ones(2, TOKENS, HEADS, dtype=torch.float64)),
          f"min {float(weights.sum(dim=(-2,-1)).min()):.6f}")
    check("initialization gives the uniform fan",
          torch.allclose(weights, torch.full_like(weights, 1.0 / (2 * POINTS))),
          f"expected {1.0 / (2 * POINTS):.4f}, got {float(weights.flatten()[0]):.4f}")
    own = default_reference_points(SHAPES)
    img_cell = own[0, 0, 0]
    check("own-modality reference is the query's own cell centre",
          bool(torch.allclose(img_cell, torch.tensor([0.5 / 36, 0.5 / 12]))),
          f"token 0 -> {img_cell.tolist()}")

# --- 4: gradients ----------------------------------------------------------
attn_f = MultiScaleDeformableAttention(
    n_embd=EMBD, n_head=HEADS, attn_pdrop=0.0, resid_pdrop=0.0,
    spatial_shapes=SHAPES, num_points=POINTS,
)
xf = torch.randn(2, TOKENS, EMBD, requires_grad=True)
attn_f(xf).square().mean().backward()
dead = [n for n, p in attn_f.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max() == 0]
check("every parameter takes a finite non-zero gradient", not dead, f"dead: {dead}")
check("input takes gradient", xf.grad is not None and float(xf.grad.abs().max()) > 0)

# --- 5: dtype and compile parity ------------------------------------------
attn_f.eval()
with torch.no_grad():
    ref32 = attn_f(xf.detach())
    if torch.cuda.is_available():
        cuda_attn = attn_f.cuda()
        xc = xf.detach().cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf = cuda_attn(xc).float()
        rel = float((bf.cpu() - ref32).abs().max() / ref32.abs().max())
        check("bf16 autocast stays close to fp32", rel < 0.05, f"max rel diff {rel:.4f}")
        compiled = torch.compile(cuda_attn)
        with torch.no_grad():
            comp = compiled(xc)
        check("torch.compile matches eager",
              torch.allclose(comp.cpu(), cuda_attn(xc).cpu(), atol=1e-4),
              f"max abs diff {float((comp.cpu() - cuda_attn(xc).cpu()).abs().max()):.2e}")
        attn_f.cpu()
    else:
        print("[SKIP] cuda checks: no GPU visible")

# --- 6: the gate's bias moves modality mass -------------------------------
with torch.no_grad():
    bias = torch.zeros(2, TOKENS, 2, dtype=torch.float64)
    bias[:, :, 1] = -20.0
    logits_b = attn.attention_weights(x).view(2, TOKENS, HEADS, 2, POINTS) + bias[:, :, None, :, None]
    w_b = F.softmax(logits_b.flatten(-2), dim=-1).view(2, TOKENS, HEADS, 2, POINTS)
    share = float(w_b[:, :, :, 1, :].sum(-1).mean())
    check("a strong negative modality bias removes that modality's mass",
          share < 1e-6, f"BEV share {share:.2e}")

# --- 7: the calibrated reference table ------------------------------------
config = load_lead_config(use_cli=False)
table, valid = fusion_geometry.calibrated_reference_points(
    config, default_reference_points(SHAPES),
    config.policy.transfuser.deformable_reference_height_meter,
)
check("calibrated table is finite", bool(torch.isfinite(table).all()))
check("calibrated table stays inside [0, 1]",
      bool((table >= 0).all() and (table <= 1).all()),
      f"range [{float(table.min()):.3f}, {float(table.max()):.3f}]")
img_cov = int(valid[:N_IMG].sum())
bev_cov = int(valid[N_IMG:].sum())
check("both modalities get some calibrated references", img_cov > 0 and bev_cov > 0,
      f"image tokens {img_cov}/{N_IMG}, BEV cells {bev_cov}/{TOKENS - N_IMG}, total {img_cov + bev_cov}/{TOKENS}")
centres = fusion_geometry.bev_cell_centres(config)
# The cells are 8 m wide, so take the nearest one to 10 m ahead rather than a window.
ahead = int((abs(centres[:, 0] - 10.0) + 2.0 * abs(centres[:, 1])).argmin())
if valid[N_IMG + ahead]:
    u = float(table[0, N_IMG + ahead, 0, 0])
    check("a BEV cell straight ahead projects near the image's horizontal centre",
          0.3 < u < 0.7, f"cell at {centres[ahead].tolist()} -> u={u:.3f}")
else:
    check("a BEV cell straight ahead is visible to some camera", False, "not covered")
bottom_centre = (IMAGE_SHAPE[0] - 1) * IMAGE_SHAPE[1] + IMAGE_SHAPE[1] // 2
if valid[bottom_centre]:
    bx, by = table[0, bottom_centre, 1].tolist()
    check("a bottom-centre image token lands in the BEV grid ahead of the ego",
          0.0 <= bx <= 1.0 and 0.3 < by < 0.7, f"-> normalized ({bx:.3f}, {by:.3f})")
else:
    check("a bottom-centre image token hits the ground plane", False, "not covered")

# --- 8: cost against the dense operator -----------------------------------
dense = SelfAttention(EMBD, HEADS, 0.0, 0.0).eval()
xs = torch.randn(8, TOKENS, EMBD)
with torch.no_grad():
    for module, name in ((dense, "dense"), (attn_f.float().eval(), "deformable")):
        module(xs)
        start = time.perf_counter()
        for _ in range(20):
            module(xs)
        print(f"       {name} CPU: {(time.perf_counter() - start) / 20 * 1000:.1f} ms/batch of 8")

print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
