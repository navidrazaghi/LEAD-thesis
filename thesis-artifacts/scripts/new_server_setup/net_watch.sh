#!/bin/bash
#
# Record what the network actually does when it drops.
#
# The outages have been diagnosed by guesswork so far -- a captive portal whose
# session expires after about eight hours was the working theory. Probing while
# the link is up does not support it: http://neverssl.com returns 200 with no
# redirect and the standard captive-portal probe returns 204, so nothing is
# intercepting traffic right now. An expired portal session would normally
# redirect plain HTTP to a login page; silent drops look more like a firewall.
#
# Which it is decides whether logging back in can be automated at all, so this
# watches until the link fails and then, at the moment of failure, records what
# each layer does: the gateway, DNS, plain HTTP (redirected or dropped?), HTTPS.
# One line per minute, so the log also dates every outage and every recovery.
#
# Costs nothing and changes nothing; it only writes to its log.
#
# Body in a function called on the last line.

main() {
	set -u
	LOG=/home/new_drive/razaghi/net_watch.log
	GATEWAY=172.20.24.1

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

	state=unknown
	while true; do
		if curl -4 -sS -o /dev/null --max-time 20 -I https://hf-mirror.com 2>/dev/null; then
			now=up
		else
			now=down
		fi

		if [ "$now" != "$state" ]; then
			say "link is $now"
			if [ "$now" = down ]; then
				# The moment of failure is the only time these answers mean
				# anything, so they are all taken here.
				say "  ping gateway: $(ping -c 2 -W 3 $GATEWAY > /dev/null 2>&1 && echo ok || echo FAIL)"
				say "  dns: $(getent hosts hf-mirror.com > /dev/null 2>&1 && echo resolves || echo FAIL)"
				say "  plain http: $(curl -4 -sS -o /dev/null -w '%{http_code} redirect=%{redirect_url}' --max-time 15 http://neverssl.com 2>&1 | tail -1)"
				say "  portal probe: $(curl -4 -sS -o /dev/null -w '%{http_code} redirect=%{redirect_url}' --max-time 15 http://www.gstatic.com/generate_204 2>&1 | tail -1)"
				say "  gateway page: $(curl -4 -sS -o /dev/null -w '%{http_code}' --max-time 10 http://$GATEWAY 2>&1 | tail -1)"
				say "  https: $(curl -4 -sS -o /dev/null -w '%{http_code}' --max-time 15 -I https://hf-mirror.com 2>&1 | tail -1)"
			fi
			state=$now
		fi
		sleep 60
	done
}

main "$@"
