#!/bin/bash
# Checks the sandbox from the agent's side: what it must reach, and what it must not.
set -u
P=http://127.0.0.1:3128
as() { sudo -n -u agent -H env -i PATH=/usr/local/bin:/usr/bin:/bin HTTPS_PROXY=$P HTTP_PROXY=$P "$@"; }
ok=0
want() { if as curl -s -o /dev/null -m 10 "$1"; then echo "reachable   $1"; else echo "BLOCKED (should reach) $1"; ok=1; fi; }
deny() { if as curl -s -o /dev/null -m 10 "$1"; then echo "REACHABLE (should be blocked) $1"; ok=1; else echo "blocked     $1"; fi; }
want https://api.anthropic.com
want https://api.openai.com
want https://api.finic.ai
deny https://runmere.finic.ai/api/v1/health
deny https://github.com
deny https://pypi.org
if as env -u HTTPS_PROXY -u HTTP_PROXY curl -s -o /dev/null -m 5 https://api.anthropic.com; then
  echo "DIRECT EGRESS WORKS (should be blocked)"; ok=1; else echo "blocked     direct egress"; fi
if as test -r /home/ubuntu; then echo "CAN READ /home/ubuntu"; ok=1; else echo "blocked     /home/ubuntu"; fi
exit $ok
