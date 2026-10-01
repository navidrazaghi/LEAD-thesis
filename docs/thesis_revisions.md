# What the thesis has to reread once the new baseline is scored

The baseline every comparison is measured against never converged; see
[baseline_convergence.md](baseline_convergence.md). This is the list of what
that touches, compiled by searching the built document rather than from memory.
Thirty-eight passages mention the baseline; these are the ones where a **number**
depends on it.

Nothing here can be acted on until the closed-loop evaluation of
`rung0_lead_recipe_post` finishes. The direction of the correction is not known
and is not assumed to be favourable.

## Numbers that change directly

| where | what |
| :--- | :--- |
| closed-loop table | `25.8 ± 4.6` / `32.9 ± 4.5` / `34.6 ± 6.0`, all three conditions |
| paired table | the differences `-15.9`, `-17.0`, `-13.2` and their intervals |
| reference-comparison table | `25.0`, `37.0`, `35.5` and all three paired differences |
| intervention table | `-0.250` and its interval |
| both abstracts | `25.8`, `32.9`, `34.6` |

## Claims whose direction may change

These are the dangerous ones: not a number to swap, but a sentence that may stop
being true.

**"rung 2a is best in all three conditions."** A stronger baseline may overturn
this in at least one condition.

**"rung 4's advantage over the baseline is resolved in all five conditions."**
Intervals tighten against a stronger baseline and may come to span zero.

**"only six and a half percent of rung 0's behaviour comes from the camera."**
This is measured on a model whose training diverged. The whole paragraph that
follows -- effect sizes read against this backdrop -- rests on it.

**"under the two unseen lidar degradations the ordering inverts and the
untouched baseline is best."** One of the two boundary findings in the abstract.

## Passages that need rewriting, not renumbering

**The attribution of the weak baseline to dataset size.** The thesis says the
baseline is weaker because of the 450-log subset. The measured cause is the
training recipe: seven rungs converge on that same subset. This diagnosis is
wrong as written and has to be corrected rather than adjusted.

**"all internal comparisons happen in the lower half of the scale."** Still true
against the reference's 90.7, but the gap changes size.

## What has to be added

A section recording the divergence itself: that the original baseline did not
converge, how it was found, what fixed it, and that the fix needed no additional
data. This is methodology, not an embarrassment, and it is the kind of thing an
examiner should read rather than discover.

If the second finding survives the full evaluation -- four open-loop measures
better and closed-loop behaviour worse -- it deserves its own subsection in the
discussion. The thesis argues on theoretical grounds that open-loop scores do not
predict closed-loop behaviour; this would be a direct measurement of exactly
that, obtained accidentally while trying to fix something else.

## Order of work

1. Let the evaluation finish (90 routes, three conditions).
2. Run `stats_audit.py` over the merged results, so every revised claim carries a
   paired interval and a Holm-corrected p-value rather than a difference of
   means.
3. Then rewrite, with measured numbers rather than predicted ones.
