#!/bin/bash
#
# Second driver: same idea as the first, running the tuned install instead.
#
# The first driver proved the retry loop works -- each attempt reached a further
# package than the last -- but at a minute and a half per attempt it was going
# to take all day. stage1_finish.sh gives uv settings that suit this link, so
# this driver should need very few attempts rather than hundreds.
#
# Order is unchanged and deliberate: the environment first, alone on the link,
# because uv cannot resume a wheel mid-download and the maps can.
#
# Body in a function called on the last line.

main() {
	set -u
	BASE=/home/new_drive/razaghi
	PY=$BASE/miniforge3/envs/lead/bin/python
	MAPS=$BASE/CARLA/standard_0916/Import/AdditionalMaps_0.9.16.tar.gz
	ATTEMPTS=300

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	env_ready() { "$PY" -c "import torch, lead" >/dev/null 2>&1; }
	maps_ready() { [ -f "$BASE/CARLA/standard_0916/.maps_imported" ]; }

	local attempt
	if env_ready; then
		say "environment already complete"
	else
		for attempt in $(seq 1 $ATTEMPTS); do
			say "environment install, attempt $attempt"
			bash "$BASE/stage1_finish.sh" >> "$BASE/stage1_chain.log" 2>&1
			if env_ready; then
				say "environment complete after $attempt attempt(s)"
				break
			fi
			say "  attempt $attempt did not finish; waiting a minute"
			sleep 60
		done
	fi

	if ! env_ready; then
		say "FATAL: the environment did not install after $ATTEMPTS attempts"
		return 1
	fi

	if maps_ready; then
		say "maps already imported"
	else
		for attempt in $(seq 1 $ATTEMPTS); do
			say "maps, attempt $attempt ($(du -cm "$MAPS".part* 2>/dev/null | tail -1 | cut -f1) MB on disk)"
			bash "$BASE/stage3_carla_v3.sh" >> "$BASE/stage3_v3.log" 2>&1
			if maps_ready; then
				say "maps complete after $attempt attempt(s)"
				break
			fi
			say "  attempt $attempt did not finish; waiting a minute"
			sleep 60
		done
	fi

	maps_ready && say "stages 1 to 3 are complete" || say "FATAL: the maps did not finish"
}

main "$@"
