#!/bin/bash
#
# Finish the environment install, with uv told to suit this link.
#
# Twenty-two attempts of the plain install each died the same way: three
# retries, forty-four seconds, connection to pypi.org timed out -- on a
# different package every time. A different package each time means the earlier
# ones did land in the cache, so the work was converging, just a minute and a
# half at a time.
#
# The four settings below address what that pattern points at. uv opens many
# connections at once by default and this link does not tolerate it, so the
# concurrency comes down to four -- the same number of streams that carried the
# 8 GB CARLA download without trouble. The timeouts go up because a request here
# can stall for a minute and still complete, and the retry count goes up because
# giving up after three is what has been ending every attempt.
#
# Picks up where the cache leaves off, so re-running costs nothing.
#
# Body in a function called on the last line.

main() {
	set -u
	BASE=/home/new_drive/razaghi
	REPO=$BASE/lead
	CONDA=$BASE/miniforge3
	PY=$CONDA/envs/lead/bin/python
	LOG=$BASE/stage1.log

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	export UV_CONCURRENT_DOWNLOADS=4
	export UV_HTTP_TIMEOUT=180
	export UV_HTTP_CONNECT_TIMEOUT=60
	export UV_HTTP_RETRIES=10

	say "installing LEAD and its dependencies (4 connections, 180s timeout, 10 retries)"
	cd "$REPO" || exit 1
	"$PY" -m uv pip install --python "$PY" -e "." >> "$LOG" 2>&1 \
		|| { say "install did not finish; tail of $LOG follows"; tail -12 "$LOG"; exit 1; }
	say "dependencies installed"

	# The three runtime settings from docs/server_setup_notes.md, which the
	# earlier script never reached because it exited at the install above.
	mkdir -p "$HOME/.local/cuda-stubs"
	if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
		ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$HOME/.local/cuda-stubs/libcuda.so"
		say "libcuda.so stub in place"
	else
		say "WARNING: libcuda.so.1 not found; torch.compile will fail until it is"
	fi

	say "fetching the resnet34 weights into the torch cache"
	"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import timm
timm.create_model("resnet34", pretrained=True)
PYEOF
	if [ $? -eq 0 ]; then say "resnet34 weights cached"; else say "WARNING: could not cache resnet34 weights"; fi

	if ! grep -q "ulimit -n 65536" "$HOME/.bashrc" 2>/dev/null; then
		{
			echo ""
			echo "# LEAD: the cache store opens one LMDB environment per log; the"
			echo "# default soft limit of 1024 makes training 50x slower without erroring."
			echo "ulimit -n 65536"
			echo "export TIMM_USE_OLD_CACHE=1"
			echo "export LIBRARY_PATH=\$HOME/.local/cuda-stubs:\${LIBRARY_PATH:-}"
			echo "export PATH=$CONDA/bin:\$PATH"
		} >> "$HOME/.bashrc"
		say "runtime settings added to ~/.bashrc"
	fi

	say "verifying"
	"$PY" - <<'PYEOF'
import torch
print(f"  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    print(f"  bf16 matmul ok: {(a @ a).shape}")
import lead
print(f"  lead imports from {lead.__file__}")
PYEOF
	say "environment finished"
}

main "$@"
