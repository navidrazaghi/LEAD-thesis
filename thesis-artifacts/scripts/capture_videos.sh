#!/bin/bash
#
# Record four demonstration videos once the running sweep is done.
#
# The whole body sits in a function called on the last line. Bash reads a
# script incrementally, remembering a byte offset, so an edit while it runs
# shifts the file under it and the next command it reads is garbage. Parsing
# the file first makes that impossible.
#
# One invocation per capture, not one sweep over four: the driver wipes its
# work directory at the start of every route, so a video only survives until
# the next route begins. With one route per invocation the files are still
# there when this script copies them out.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	KEEP=~/videos
	SCRATCH=outputs/eval_scratch
	LOG=~/capture_videos.log
	mkdir -p "$KEEP"

	say() { echo "[$(date +%H:%M:%S)] $*"; }

	say "waiting for the sweep to finish"
	while pgrep -f "run_evaluation.py --models rung0_lead" >/dev/null; do
		sleep 60
	done
	say "the sweep has exited"
	sleep 20

	# route, condition, label
	CAPTURES=(
		"11715.xml none:0    stall"
		"24071.xml none:0    intact"
		"24071.xml camera:1.0 camera_destroyed"
		"24071.xml lidar:1.0  lidar_destroyed"
	)

	for entry in "${CAPTURES[@]}"; do
		set -- $entry
		route=$1; condition=$2; label=$3

		say "capturing $label  ($route under $condition)"
		printf "%s\n" "$route" > /tmp/one_route.txt
		rm -rf "$SCRATCH"

		~/miniconda3/envs/lead/bin/python scripts/common/run_evaluation.py \
			--models rung0_lead=outputs/rung0_lead_recipe_post \
			--routes /tmp/one_route.txt \
			--conditions "$condition" \
			--out "$KEEP/capture_$label.csv" \
			--config evaluation.produce_demo_video=true \
			         evaluation.produce_grid_video=true \
			>> "$LOG" 2>&1

		found=0
		for mp4 in "$SCRATCH"/*.mp4; do
			[ -e "$mp4" ] || continue
			found=1
			base=$(basename "$mp4" .mp4)
			cp "$mp4" "$KEEP/${label}_${base}.mp4"
			say "  kept ${label}_${base}.mp4  ($(du -h "$mp4" | cut -f1))"
		done
		[ "$found" = 0 ] && say "  WARNING: no mp4 produced for $label"
	done

	say "compressing for transfer"
	for mp4 in "$KEEP"/*.mp4; do
		case "$mp4" in *_small.mp4) continue;; esac
		small="${mp4%.mp4}_small.mp4"
		[ -e "$small" ] && continue
		ffmpeg -nostdin -loglevel error -i "$mp4" \
			-vf "scale=1280:-2" -crf 28 -preset slow -an "$small" </dev/null
		say "  $(basename "$small")  $(du -h "$mp4" | cut -f1) -> $(du -h "$small" | cut -f1)"
	done

	say "done"
	ls -lh "$KEEP"/*.mp4 2>/dev/null | awk "{print \$5, \$9}"
}

main "$@"
