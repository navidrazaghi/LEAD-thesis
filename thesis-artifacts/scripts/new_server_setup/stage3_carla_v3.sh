#!/bin/bash
#
# Stage 3, third attempt: CARLA 0.9.16 and its additional maps.
#
# What the two earlier attempts established. The URL the repository's setup
# script uses (tiny.carla.org) redirects to a CDN that will not complete an SSL
# handshake from this network, so the files come from the Backblaze origin
# behind it. And curl's --retry cannot be combined with an explicit byte range
# written into an appended file: a retry re-requests the range from its first
# byte and the append leaves the earlier bytes in place, so every drop silently
# duplicates a prefix. The second attempt lost that way -- 9.36 GB written for
# an 8.35 GB file.
#
# So --retry is gone here. Retrying is the outer loop's job: it measures what
# each part already holds and asks only for the bytes still missing, which is
# correct whether the last attempt ended cleanly or was cut off mid-stream.
# After the parts are joined, gzip -t checks a CRC over the whole archive before
# anything is extracted.
#
# Safe to re-run at any point: every step is skipped when its result is there.
#
# Body in a function called on the last line.

main() {
	set -u
	BASE=/home/new_drive/razaghi
	TARGET=$BASE/CARLA/standard_0916
	ORIGIN=https://carla-releases.s3.us-east-005.backblazeb2.com/Linux
	LOG=$BASE/stage3.log
	PARTS=4
	ROUNDS=40

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	mkdir -p "$TARGET/Import"

	# Download one file as PARTS byte ranges at once, then join and verify them.
	parallel_fetch() {
		local url=$1 out=$2 name=$3
		if [ -f "$out.done" ]; then
			say "$name already complete: $(du -h "$out" | cut -f1)"
			return 0
		fi
		local total
		total=$(curl -sSI --max-time 60 "$url" | awk '/[Cc]ontent-[Ll]ength/{print $2}' | tr -d '\r')
		if [ -z "$total" ]; then
			say "FATAL: no content-length for $name"
			return 1
		fi
		say "$name: $((total / 1000000)) MB in $PARTS parts"
		local chunk=$((total / PARTS))

		part_want() {  # index -> how many bytes that part's range holds
			local i=$1 start=$((i * chunk)) end
			end=$((start + chunk - 1))
			[ "$i" -eq $((PARTS - 1)) ] && end=$((total - 1))
			echo $((end - start + 1))
		}
		have_total() {
			local i sum=0
			for i in $(seq 0 $((PARTS - 1))); do
				[ -f "$out.part$i" ] && sum=$((sum + $(stat -c %s "$out.part$i")))
			done
			echo "$sum"
		}

		local round
		for round in $(seq 1 $ROUNDS); do
			local pids=() i
			for i in $(seq 0 $((PARTS - 1))); do
				local start=$((i * chunk))
				local want
				want=$(part_want "$i")
				local part="$out.part$i"
				local have=0
				[ -f "$part" ] && have=$(stat -c %s "$part")
				if [ "$have" -ge "$want" ]; then
					continue
				fi
				# No --retry: the range asked for is exactly what is missing,
				# and a drop simply leaves this part short for the next round.
				curl -fsS -r "$((start + have))-$((start + want - 1))" \
					--max-time 7200 "$url" >> "$part" 2>> "$LOG" &
				pids+=($!)
			done
			if [ ${#pids[@]} -eq 0 ]; then
				break
			fi
			wait "${pids[@]}"
			say "  round $round: $(($(have_total) / 1000000)) of $((total / 1000000)) MB"
			[ "$(have_total)" -ge "$total" ] && break
			sleep 10
		done

		# Each part must now measure its range exactly. A longer one would mean
		# the duplication bug is back, and joining it would produce a corrupt
		# archive, so stop rather than extract something broken.
		local i
		for i in $(seq 0 $((PARTS - 1))); do
			local want size
			want=$(part_want "$i")
			size=$(stat -c %s "$out.part$i" 2>/dev/null || echo 0)
			if [ "$size" -ne "$want" ]; then
				say "FATAL: part$i is $size bytes, its range is $want; re-run to continue"
				return 1
			fi
		done

		say "joining $name"
		cat "$out".part* > "$out" || { say "FATAL: join failed"; return 1; }
		local size
		size=$(stat -c %s "$out")
		if [ "$size" -ne "$total" ]; then
			say "FATAL: $name joined to $size bytes, expected $total"
			return 1
		fi
		say "verifying the archive (a CRC over all $((total / 1000000)) MB; this takes a few minutes)"
		if ! gzip -t "$out" 2>> "$LOG"; then
			say "FATAL: $name fails its CRC; the parts are not sound"
			return 1
		fi
		rm -f "$out".part*
		touch "$out.done"
		say "$name complete and verified: $(du -h "$out" | cut -f1)"
	}

	cd "$TARGET" || exit 1
	parallel_fetch "$ORIGIN/CARLA_0.9.16.tar.gz" "$TARGET/CARLA_0916.tar.gz" "CARLA 0.9.16" || exit 1

	if [ ! -x "$TARGET/CarlaUE4.sh" ]; then
		say "extracting CARLA"
		tar -xzf "$TARGET/CARLA_0916.tar.gz" -C "$TARGET" >> "$LOG" 2>&1 \
			|| { say "FATAL: extract failed"; exit 1; }
	fi
	say "CARLA extracted"

	parallel_fetch "$ORIGIN/AdditionalMaps_0.9.16.tar.gz" \
		"$TARGET/Import/AdditionalMaps_0.9.16.tar.gz" "additional maps" || exit 1

	if [ ! -f "$TARGET/.maps_imported" ]; then
		say "importing the additional maps; this rewrites the package and takes a while"
		cd "$TARGET" && bash ImportAssets.sh >> "$LOG" 2>&1 && touch "$TARGET/.maps_imported"
	fi
	say "maps imported"

	say "installed size: $(du -sh "$TARGET" | cut -f1)"
	say "stage 3 done: $TARGET"
}

main "$@"
