"""A per-token gain on the attention's contribution to the residual stream.

The modality gate decides *which* modality a query reads from. It cannot decide
*how much* of what it read enters the token, because the block adds the
attention output to a residual stream whose magnitude the gate does not touch.
Measuring both factors shows what that costs: the camera's share of a BEV
query's attention mass reaches 0.50 in the gated rung, but the attention
contributes only about 0.40 of what leaves the block, and the product -- 0.20 --
is what the intervention test actually measures as causal reliance. The gate has
pushed the first factor near its useful range and has no access to the second.

This module supplies the second. It predicts one scalar per token, passes it
through a sigmoid scaled to ``2 * max_gain`` so a neutral prediction reproduces
the unmodified block exactly, and multiplies the attention output by it before
the residual add.

Two behaviours follow that the gate cannot provide. Where one modality is
unreliable the gate can shift weight to the other, and the gain can additionally
raise how much of that shifted read survives into the token. Where *both*
modalities are unreliable the gate is powerless -- reweighting between two bad
readings gives a bad reading -- while the gain can shut the contribution down
and leave the token on the content it already carried, which is the behaviour
adverse weather calls for and which the thesis lists as an unaddressed
limitation.

Zero-initialised in the same sense the gate is: the projection starts at zero,
so the sigmoid starts at one half, so the gain starts at exactly one and a model
carrying this module begins where the model without it begins.
"""

import jaxtyping as jt
import torch
from torch import nn


class ResidualGain(nn.Module):
    """Scales the attention output before it joins the residual stream."""

    def __init__(self, n_embd: int, max_gain: float = 2.0) -> None:
        """Initialise the gain as an exact no-op.

        Args:
            n_embd: Embedding dimension of the fusion tokens.
            max_gain: Upper bound of the gain; the lower bound is zero and the
                starting value is one, which requires this to be two.
        """
        super().__init__()
        self.head = nn.Linear(n_embd, 1)
        self.max_gain = max_gain
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Zero the projection so the gain starts at one.

        Must run after any generic initialisation that would overwrite it,
        otherwise the model does not start from the ungained one.
        """
        nn.init.constant_(self.head.weight, 0.0)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(
        self,
        x: jt.Float[torch.Tensor, "B T C"],
    ) -> jt.Float[torch.Tensor, "B T 1"]:
        """Predict how much of the attention output each token should take.

        Args:
            x: The fusion tokens, already normalised by the block.

        Returns:
            One gain per token, in ``[0, max_gain]``, starting at one.
        """
        return torch.sigmoid(self.head(x)) * self.max_gain
