#!/bin/bash
#
# Stage 1 and 2 on the new machine: the Python environment, then the three
# runtime settings this project has learned the hard way.
#
# Everything lands on the big disk. repo.anaconda.com answers 403 from this
# network, so the environment comes from Miniforge on GitHub, which does not.
#
# Safe to re-run: each step is skipped when its result is already there.
#
# Body in a function called on the last line, so an edit mid-run cannot corrupt
# what bash is reading.

main() {
	set -u
	BASE=/home/new_drive/razaghi
	REPO=$BASE/lead
	CONDA=$BASE/miniforge3
	LOG=$BASE/stage1.log

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	mkdir -p "$BASE"
	cd "$BASE" || exit 1

	# --- 1. Miniforge -------------------------------------------------------
	if [ ! -x "$CONDA/bin/conda" ]; then
		say "downloading Miniforge"
		curl -fsSL -o /tmp/miniforge.sh \
			https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
			>> "$LOG" 2>&1 || { say "FATAL: Miniforge download failed"; exit 1; }
		say "installing Miniforge into $CONDA"
		bash /tmp/miniforge.sh -b -p "$CONDA" >> "$LOG" 2>&1 || { say "FATAL: Miniforge install failed"; exit 1; }
	fi
	say "conda: $("$CONDA/bin/conda" --version)"

	# --- 2. the lead environment -------------------------------------------
	if [ ! -x "$CONDA/envs/lead/bin/python" ]; then
		say "creating the lead environment, python 3.10"
		"$CONDA/bin/conda" create -y -n lead python=3.10 >> "$LOG" 2>&1 \
			|| { say "FATAL: env create failed"; exit 1; }
	fi
	PY=$CONDA/envs/lead/bin/python
	say "python: $($PY -V 2>&1)"

	# --- 3. the project and its dependencies -------------------------------
	say "installing uv"
	"$PY" -m pip install -q uv >> "$LOG" 2>&1 || { say "FATAL: uv install failed"; exit 1; }
	say "installing LEAD and its dependencies (torch 2.8 among them; this is the long one)"
	cd "$REPO" || exit 1
	"$PY" -m uv pip install --python "$PY" -e "." --reinstall-package lead >> "$LOG" 2>&1 \
		|| { say "FATAL: dependency install failed; tail of $LOG follows"; tail -25 "$LOG"; exit 1; }
	say "dependencies installed"

	# --- 4. the three runtime settings from docs/server_setup_notes.md -----
	# torch.compile links Triton kernels against an unversioned libcuda.so that
	# the driver does not ship; without this the default training command dies.
	mkdir -p "$HOME/.local/cuda-stubs"
	if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
		ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$HOME/.local/cuda-stubs/libcuda.so"
		say "libcuda.so stub in place"
	else
		say "WARNING: libcuda.so.1 not found; torch.compile will fail until it is"
	fi

	# The backbone builds resnet34 with pretrained=True. Fetching the weights
	# once now means training never depends on the Hub being up.
	say "fetching the resnet34 weights into the torch cache"
	"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import timm
timm.create_model("resnet34", pretrained=True)
print("resnet34 weights cached")
PYEOF
	if [ $? -eq 0 ]; then say "resnet34 weights cached"; else say "WARNING: could not cache resnet34 weights"; fi

	# ulimit -n is per shell, so it belongs in the shell profile as well as in
	# every run script: the default 1024 makes training 50x slower, silently.
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

	# --- 5. verification ----------------------------------------------------
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
	say "stage 1 and 2 done"
}

main "$@"
