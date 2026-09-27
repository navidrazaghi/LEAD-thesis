#!/bin/bash
#
# A remote desktop for this machine: XFCE, served by xrdp, reachable only
# through an SSH tunnel.
#
# The machine boots to graphical.target but has no graphical stack at all -- no
# GNOME, no gdm3, no Xorg -- so the desktop itself has to be installed, not
# just the remote tool. XFCE rather than GNOME because GNOME under xrdp on 24.04
# is heavy and unreliable, and xrdp rather than VNC because Windows already has
# the client built in.
#
# xrdp is bound to 127.0.0.1. This is a shared machine on the university
# network; nothing here should listen beyond it. From Windows, a tunnel carries
# the connection:
#
#     ssh -L 13389:127.0.0.1:3389 -N razaghi@172.20.27.6
#     then Remote Desktop Connection to 127.0.0.1:13389
#
# --no-install-recommends keeps a display manager (lightdm) out: nothing
# should start a graphical login on the console or claim the GPU at boot.
# xrdp starts its own software-rendered X server per session, which leaves the
# A100 to CUDA and CARLA.
#
# The apt downloads wait for the dataset download to finish, so the two do not
# fight over a link that already drops. Needs root; run it once with sudo.
#
# Body in a function called on the last line.

main() {
	set -u
	BASE=/home/new_drive/razaghi
	USER_NAME=razaghi
	USER_HOME=/home/$USER_NAME

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	if [ "$(id -u)" -ne 0 ]; then
		echo "run this with sudo"
		exit 1
	fi

	# --- 1. wait for the dataset download ------------------------------------
	while pgrep -f '[f]etch_stage4.py' > /dev/null; do
		say "dataset download still running; checking again in 5 minutes"
		sleep 300
	done
	say "no dataset download running; installing"

	# --- 2. packages ---------------------------------------------------------
	# Retries and a lock timeout because this link drops and unattended-upgrades
	# may hold the dpkg lock.
	APT=(apt-get -y -o Acquire::Retries=10 -o DPkg::Lock::Timeout=900)
	"${APT[@]}" update || { say "FATAL: apt update failed"; exit 1; }
	DEBIAN_FRONTEND=noninteractive "${APT[@]}" install --no-install-recommends \
		xfce4 xfce4-terminal dbus-x11 xrdp xorgxrdp \
		|| { say "FATAL: install failed"; exit 1; }
	say "installed: $(dpkg-query -W -f='${Package} ${Version}, ' xfce4 xrdp xorgxrdp)"

	# --- 3. xrdp: localhost only ---------------------------------------------
	INI=/etc/xrdp/xrdp.ini
	[ -f "$INI.orig" ] || cp "$INI" "$INI.orig"
	sed -i -E 's|^port=.*|port=tcp://127.0.0.1:3389|' "$INI"
	say "xrdp.ini: $(grep -E '^port=' "$INI" | head -1)"

	# xrdp reads its TLS key from /etc/ssl/private, which ssl-cert owns.
	adduser xrdp ssl-cert > /dev/null 2>&1 || true

	# --- 4. the session this user gets --------------------------------------
	echo "startxfce4" > "$USER_HOME/.xsession"
	chown "$USER_NAME:$USER_NAME" "$USER_HOME/.xsession"
	say "$USER_HOME/.xsession -> startxfce4"

	# --- 5. start and verify -------------------------------------------------
	systemctl enable xrdp > /dev/null 2>&1
	systemctl restart xrdp
	sleep 3
	say "xrdp: $(systemctl is-active xrdp)"
	LISTEN=$(ss -ltn | grep ':3389 ' || true)
	say "listening: ${LISTEN:-nothing on 3389}"
	if echo "$LISTEN" | grep -q '127.0.0.1:3389'; then
		say "done. From Windows: ssh -L 13389:127.0.0.1:3389 -N razaghi@172.20.27.6"
		say "then Remote Desktop Connection to 127.0.0.1:13389, user razaghi"
	else
		say "WARNING: xrdp is not listening on 127.0.0.1:3389 as intended; check it before using"
	fi
}

main "$@"
