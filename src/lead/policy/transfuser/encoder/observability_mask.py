"""Replace an unreliable token with a learned prior instead of down-weighting it.

The gate and this module read the same signal and disagree about what to do
with it. Measured against the expert's labels on logs it never trained on, the
head that produces that signal works: mean absolute error 0.160 against a
constant predictor's 0.232 for the camera, and 0.193 against 0.369 for LiDAR.
The detector is not the problem.

What the gate does with it is a bias added to the operator's modality logits
before the softmax, so a query reads less from the damaged modality and more
from the other. That is a reweighting, and it leaves the damaged features in
place: they still travel the branch trunk into the planning decoder, which the
gate never touches. It also has nothing to offer when *both* modalities are
degraded, because reweighting between two bad readings still gives a bad
reading -- and that case is adverse weather, which is the condition the thesis
set out to improve and the one its numbers are weakest on.

The shape of the failure supports that reading. Under LiDAR destruction the
corrected baseline scores 16.26, far below the 43.08 it manages with the camera
destroyed instead. A model merely short of information would fall towards its
single-sensor score; one that falls well below it is being actively misled.

So this substitutes rather than reweights. Where the head says a token's own
modality is unobservable, the token is moved towards a learned per-modality
prior -- an "unknown", which the fusion transformer can treat as absent
information rather than as a confident reading that happens to be wrong. The
approach is the one the missing-modality literature converges on: masked
modality training with a learnable token, which supplies a modality-agnostic
prior instead of leaving the trunk to interpret noise.

Nothing here is a new detector. The head, its targets and its loss are the
gate's, unchanged, so a run carrying this module differs from a gated run in
one thing: what the signal is used for.
"""

import jaxtyping as jt
import torch
from torch import nn

# Starting value of the substitution strength, as a logit: sigmoid(-6) is about
# 0.0025, so a freshly built model admits a quarter of a per cent of the prior
# and begins, for practical purposes, where the unmasked model begins. Measured
# on a forward pass, the feature maps move by 5e-3 and 8e-2 against a control
# whose trained-mask range is 2.9 and 12.4 -- under a per cent of what the
# mechanism can do. A hard zero would be an exact no-op and also a dead end,
# because clamping at zero kills the gradient that has to lift it.
_INITIAL_STRENGTH_LOGIT = -6.0


class ObservabilityMask(nn.Module):
    """Moves tokens towards a learned prior where their own modality is unreliable."""

    # Declared so the type checker resolves the registered buffer here rather
    # than through nn.Module.__getattr__, which types every attribute as a union.
    token_level: torch.Tensor

    def __init__(
        self,
        n_embd: int,
        spatial_shapes: tuple[tuple[int, int], ...],
    ) -> None:
        """Build one prior per modality and the strength that admits it.

        Args:
            n_embd: Embedding dimension of the fusion tokens.
            spatial_shapes: ``(height, width)`` of each modality's token grid,
                in the order the tokens are concatenated. This is what says
                which modality each token belongs to.
        """
        super().__init__()
        self.num_levels = len(spatial_shapes)
        self.prior = nn.Parameter(torch.zeros(self.num_levels, n_embd))
        self.strength = nn.Parameter(torch.tensor(_INITIAL_STRENGTH_LOGIT))

        # Which modality each token is a piece of. The operator packs the
        # grids in the order spatial_shapes gives them, so this is the same
        # ordering the gate's modality axis uses; reading it the other way
        # round would substitute the intact sensor and look like a null result.
        levels = torch.cat(
            [
                torch.full((height * width,), level, dtype=torch.long)
                for level, (height, width) in enumerate(spatial_shapes)
            ],
        )
        self.register_buffer("token_level", levels, persistent=False)

    def forward(
        self,
        x: jt.Float[torch.Tensor, "B T C"],
        gate_logits: jt.Float[torch.Tensor, "B T L"],
    ) -> jt.Float[torch.Tensor, "B T C"]:
        """Move each token towards its modality's prior in proportion to doubt.

        Args:
            x: The fusion tokens, already normalized by the block.
            gate_logits: The head's per-token, per-modality observability
                logits -- the same tensor the gate would add to the operator's
                modality logits.

        Returns:
            The tokens, with unreliable ones pulled towards their prior.
        """
        batch, tokens = x.shape[0], x.shape[1]
        level = self.token_level.view(1, tokens, 1).expand(batch, tokens, 1)

        # A token's own modality is the one it is a piece of: an image token
        # asks how observable the camera is where it looks, a BEV token asks
        # about the LiDAR. The other column is what the gate used and is not
        # read here.
        own = torch.gather(gate_logits, 2, level)
        doubt = torch.sigmoid(-own)

        strength = torch.sigmoid(self.strength)
        prior = self.prior.index_select(0, self.token_level).unsqueeze(0)
        return x + strength * doubt * (prior.to(x.dtype) - x)
