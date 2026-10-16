#!/bin/bash
#
# The weather campaign the thesis says it did not run.
#
# WHY THIS RUN EXISTS
#
# Chapter five states the gap in its own words: the training data already
# contains rain, fog and night, and "what was not done is using weather
# conditions as an evaluation axis, so the gap reported there is an experiment
# gap and not a data gap." That sentence is honest and it is also the first
# thing an examiner will pull on, because the experiment was affordable and the
# route set for it has been committed since it was chosen.
#
# It was in fact started once. ~/weather_eval.log shows a 160-run matrix that
# stopped at run 16, leaving results/weather_closed_loop.csv with four routes
# of the forty, all of them Town12. Four routes cannot carry an
# out-of-distribution claim, which is presumably why the thesis declares the
# gap rather than quoting them.
#
# WHAT MAKES THIS SET THE RIGHT ONE
#
# src/lead/routes/eval_sets/weather.txt was built for exactly this question and
# is disjoint from the thirty degradation routes -- checked, zero overlap. Its
# selector's own docstring says why it exists: training saw 0.6 per cent rain
# and 0.2 per cent night against roughly half of Bench2Drive, so these routes
# are a genuine out-of-distribution test that needs no synthetic damage at all.
# Hence one condition, none:0. Adding synthetic damage on top of real weather
# would leave nobody able to say which of the two moved the score.
#
# WHICH MODELS, AND WHY THESE
#
# Only models trained at the published recipe: 31 epochs of pretrain, 31 of
# post-train, effective batch 64. Verified from each run's own config.yaml
# rather than assumed. The ladder's own rungs, at effective batch 8 for 10
# epochs, are deliberately absent for the same reason they are absent from
# every other recipe-era comparison.
#
# ref0 is the exception and is not a rung. It is the upstream authors'
# published checkpoint, at a true batch 64 with no accumulation, and its config
# does not record which logs it trained on -- so a gap against it mixes
# architecture with training-set size and must be read that way. It is here
# because without an external anchor the weather numbers have nothing to be
# read against, and because on the four routes already scored it took 83.55
# where our models took 14 to 29. If that gap survives forty routes it is a
# finding in its own right.
#
# WHAT THIS WRITES, AND WHAT IT LEAVES ALONE
#
# A new file. results/weather_closed_loop.csv holds the sixteen rows of the
# abandoned campaign, and those rows are the evidence for what the ladder's
# regime does under weather; overwriting them would destroy a comparison that
# cannot be rebuilt without retraining the old rungs.
#
# The body sits in a function called on the last line. Bash reads a script by
# byte offset, so editing one while it runs feeds it garbage from the shift;
# parsing the whole file first makes that impossible.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	CSV=results/weather_recipe.csv
	ROUTES=src/lead/routes/eval_sets/weather.txt

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LEAD_RUNTIME_TYPE_CHECKING=false
	export TIMM_USE_OLD_CACHE=1
	export WANDB_MODE=offline
	ulimit -n 65536

	wait_for_our_gpu_to_clear || exit 1
	check_disk || exit 1

	ROUTE_COUNT=$(wc -l < "$ROUTES")
	say "route set: $ROUTE_COUNT weather routes, scored at zero synthetic damage"

	# Include a model only if its checkpoint is on disk. rung 3 is produced by
	# the run before this one, and a run that failed must not take the campaign
	# down with it -- the other four still answer the question.
	MODELS=()
	for pair in \
		"post31=outputs/rung0_lead_recipe_post31" \
		"rung2a_recipe=outputs/rung2a_lead_recipe_post" \
		"rung2ad_recipe=outputs/rung2ad_lead_recipe_post" \
		"rung3_recipe=outputs/rung3_lead_recipe_post" \
		"ref0=reference/seed0"
	do
		name=${pair%%=*}
		path=${pair#*=}
		if [ -n "$(ls "$path"/model_*.pth 2>/dev/null)" ]; then
			MODELS+=("$pair")
			say "  including $name"
		else
			say "  SKIPPING $name: no checkpoint at $path"
		fi
	done

	if [ ${#MODELS[@]} -eq 0 ]; then
		say "FATAL: no checkpoints found; nothing to score"
		exit 1
	fi
	say "${#MODELS[@]} models x $ROUTE_COUNT routes = $(( ${#MODELS[@]} * ROUTE_COUNT )) runs"

	# --out keeps the rows already in the file and skips them, so an
	# interrupted campaign resumes rather than restarting.
	$PY scripts/common/run_evaluation.py \
		--models "${MODELS[@]}" \
		--routes "$ROUTES" \
		--conditions none:0 \
		--out "$CSV" \
		>> ~/weather_recipe_eval.log 2>&1

	if [ ! -s "$CSV" ]; then
		say "FATAL: no rows written; see ~/weather_recipe_eval.log"
		exit 1
	fi
	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"

	carry_out
	say "read against results/weather_closed_loop.csv, which holds the same"
	say "question asked of the ladder's regime on four of these routes."
}

# Copy the result into the tracked tree and commit it locally.
#
# results/ is git-ignored, so until this runs the campaign exists on one disk.
# The commit is local: pushing needs credentials a background script must not
# hold.
carry_out() {
	local repo="$HOME/LEAD/lead"
	local tracked="thesis-artifacts/results/weather_recipe.csv"
	cp "$repo/results/weather_recipe.csv" "$repo/$tracked" || return 0
	if git -C "$repo" add "$tracked" &&
		git -C "$repo" commit -q -m "Carry the weather campaign out of the machine

Written by run_weather_recipe.sh when the campaign finished. results/ is
ignored, so until this commit these rows existed on one disk only." \
			-- "$tracked"; then
		say "committed $tracked"
		say "NOT pushed. Push with: git -C ~/LEAD/lead push thesis robust-deployment"
	else
		say "commit failed or nothing changed; the file is in $tracked either way"
	fi
}

# Wait for our own processes to leave the GPU, then start. Only ours: the card
# is shared, and user omati has held about 1.2 GB with a YOLO process for over
# a day, so a guard that refuses on any compute app would refuse forever.
wait_for_our_gpu_to_clear() {
	local waited=0
	local limit=3600
	while true; do
		local busy=""
		for gpu_pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
			if [ "$(ps -o user= -p "$gpu_pid" 2>/dev/null | tr -d ' ')" = "$(id -un)" ]; then
				busy="$gpu_pid"
				break
			fi
		done
		[ -z "$busy" ] && break
		if [ "$waited" -ge "$limit" ]; then
			say "FATAL: pid $busy of ours has held the GPU for ${limit}s; not starting"
			ps -o pid=,etime=,cmd= -p "$busy" | cut -c1-120
			return 1
		fi
		[ "$waited" = 0 ] && say "waiting for our pid $busy to leave the GPU"
		sleep 60
		waited=$(( waited + 60 ))
	done
	say "GPU is ours"
	return 0
}

# This campaign trains nothing, so it only needs room for evaluation logs.
check_disk() {
	local free_gb
	free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
	if [ "$free_gb" -lt 5 ]; then
		say "FATAL: only ${free_gb}G free"
		return 1
	fi
	say "disk: ${free_gb}G free"
	return 0
}

main "$@"
