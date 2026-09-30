#!/bin/bash
# Execution latency: the long-window control, then the curriculum on top of it.
#
# WHY THIS IS THE DIRECTION
#
# The caution governor failed for a reason that was about the conditions, not
# the mechanism: where the fault was real nothing drove, and where it was
# detectable the policy already coped. Latency is the missing regime. The car
# still drives; it just acts on a plan computed for a moment it has already
# left.
#
# The literature agrees it is open. Bench2Drive-Robust (2026) calls itself the
# first device-centric robustness benchmark for closed-loop driving under
# deployment perturbations and says latency and ego-state error remain little
# studied -- and it is a benchmark, not a method. The correction itself is
# proven elsewhere: Delay-Aware Diffusion Policy reports success rates far more
# robust to delay from exactly this recipe, on manipulation rather than driving.
#
# TWO RUNGS, BECAUSE THE WINDOW IS NOT FREE
#
# The curriculum needs future_ego_pose_extra_ticks, which widens how much future
# every scene must supply and so drops the scenes too near the end of a log.
# Measured here, that is 900 of 58,989 samples -- 1.53%. Small, and not nothing:
# a latency rung compared straight against rung2a_dense_curriculum would differ
# in its training set as well as its augmentation, and the comparison could not
# say which one moved the result.
#
# So the first rung carries the wider window with the curriculum switched off.
# It loses exactly the same 900 samples and is otherwise the dense control. The
# second adds the shift and nothing else.
#
# WHAT THE SHIFT DOES
#
# The label is re-anchored onto the pose the plan will be executed at, rather
# than the observation being delayed. The observation route was refused
# deliberately: the lidar raster is cached and fingerprinted by its tick ages,
# so delaying a sweep forces a full re-cache. Moving the label does the same job
# from the other side and leaves the cache alone.
#
# Verified before this script existed: with the window at 10 ticks and the
# probability at 1.0, 36 of 86 sampled labels change and the predicted horizon
# stays 8x2, so the head width and every checkpoint survive.
#
# ORDERING
#
# rung6 is already waiting on the training lock, and flock promises exclusion
# rather than order, so taking the lock here would be a coin toss with it. This
# waits for rung6's own final checkpoint to appear first, which is deterministic,
# and only then joins the queue.
#
# Everything runs inside a function called on the last line: bash reads a script
# incrementally, so editing this file while it runs would corrupt it.

set -u

main() {
  cd "$HOME/LEAD/lead" || exit 1
  ulimit -n 65536

  local out="$HOME/LEAD/lead/outputs"
  local final="model_0009.pth"
  local control=rung7a_longwindow
  local latency=rung7b_latency

  # 10 ticks at the 5-tick planning stride is two shift steps, 0.25 s and
  # 0.50 s, which brackets the 50-200 ms range Bench2Drive-Robust injects.
  local window=10
  local probability=0.5

  echo "[$(date +%H:%M:%S)] waiting for rung6 to finish before joining the lock queue"
  while [ ! -f "$out/rung6_residual_gain_post/$final" ]; do
    # Anchored at both ends and tolerant of an absolute path: how a driver is
    # launched is not something this should depend on. Matching the exact
    # relative command line meant that starting rung6 by absolute path -- which
    # is how it had to be started, after a shell precedence bug ate the relative
    # one -- made it invisible here, and this went ahead without it.
    if ! ps -eo args --no-headers \
         | grep -qE "^bash (.*/)?scripts/common/run_residual_gain[.]sh$" \
       && [ ! -d "$out/rung6_residual_gain" ]; then
      echo "[$(date +%H:%M:%S)] rung6 is neither queued nor started; going ahead without it"
      break
    fi
    sleep 600
  done

  echo "[$(date +%H:%M:%S)] waiting for the training lock"
  exec 9>"$HOME/.lead_training.lock"
  flock -w 604800 9 || { echo "timed out waiting for the lock"; exit 200; }
  echo "[$(date +%H:%M:%S)] lock acquired"

  stage() {
    local name=$1; shift
    if [ -f "$out/$name/$final" ]; then
      echo "[$(date +%H:%M:%S)] SKIP $name (already finished)"
      return 0
    fi
    echo "[$(date +%H:%M:%S)] START $name"
    bash scripts/common/run_rung.sh "$name" "$@" || return 1
    [ -f "$out/$name/$final" ] || {
      echo "[$(date +%H:%M:%S)] $name ended without $final"
      return 1
    }
    echo "[$(date +%H:%M:%S)] DONE $name"
  }

  rung() {
    local name=$1; shift
    stage "$name" "$@" || return 1
    # resume_from_last_checkpoint also decides whether the state-dict load is
    # strict, and it must be false here: the pretrain has no planning decoder,
    # so a strict load would refuse the weights this stage exists to extend.
    stage "${name}_post" "$@" \
      policy.transfuser.use_planning_decoder=true \
      training.experiment.resume_from_last_checkpoint=false \
      training.experiment.initial_weights_file="$out/$name/$final" \
      || return 1
  }

  # The control: the wider read window, the curriculum inert. One change against
  # rung2a_dense_curriculum, and it is the change that costs the 900 samples.
  rung "$control" \
    policy.transfuser.future_ego_pose_extra_ticks="$window" \
    training.data.latency_curriculum_probability=0.0 \
    training.data.use_sensor_degradation=true \
    || { echo "long-window control failed"; exit 1; }

  # One change against the control: the shift itself.
  rung "$latency" \
    policy.transfuser.future_ego_pose_extra_ticks="$window" \
    training.data.latency_curriculum_probability="$probability" \
    training.data.use_sensor_degradation=true \
    || { echo "latency rung failed"; exit 1; }

  echo "[$(date +%H:%M:%S)] latency line complete; evaluate ${latency}_post against ${control}_post"
}

main "$@"
