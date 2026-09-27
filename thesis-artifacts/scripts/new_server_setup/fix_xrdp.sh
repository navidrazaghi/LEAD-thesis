#!/bin/bash
#
# Undo the over-broad edit in setup_remote_desktop.sh and make the one it meant.
#
# That script ran sed 's|^port=.*|...|' over the whole of xrdp.ini, meant for the
# listening port in [Globals]. It also rewrote the port= line of four session
# sections, among them [Xorg]'s port=-1 ("let sesman choose"), so after a
# successful login xrdp tried to reach its own listener instead of starting a
# session, and hung on its background colour. The original was kept as
# xrdp.ini.orig; this restores it and changes only the first port= line, which
# is [Globals].
#
# Body in a function called on the last line.

main() {
	set -u
	INI=/etc/xrdp/xrdp.ini
	if [ "$(id -u)" -ne 0 ]; then echo "run this with sudo"; exit 1; fi
	[ -f "$INI.orig" ] || { echo "FATAL: no $INI.orig to restore from"; exit 1; }

	cp "$INI.orig" "$INI"
	# 0,/re/ addresses up to the first match only: [Globals]'s line and no other.
	sed -i -E '0,/^port=/s|^port=.*|port=tcp://127.0.0.1:3389|' "$INI"

	echo "port= lines now:"
	awk '/^\[/{s=$0} /^port=/{print "  " s " -> " $0}' "$INI"

	systemctl restart xrdp
	sleep 2
	echo "xrdp: $(systemctl is-active xrdp), sesman: $(systemctl is-active xrdp-sesman)"
	ss -ltn | grep ':3389 ' | sed 's/^/  /'
}

main "$@"
