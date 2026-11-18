#!/bin/bash
# Checks the sandbox from the players' side: each player reaches only loopback (and through the proxy only the model
# providers), reads only its own folder, and has no credential on a command line. Run as `ubuntu` after a game has made the players' folders (or with none: the folder checks
# then pass vacuously).
set -u
ok=0
as() { local u=$1; shift; sudo -n -u "$u" -H /usr/local/bin/arena-exec /dev/stdin "$@" <<<'{"PATH": "/usr/local/bin:/usr/bin:/bin"}'; }
bad() { echo "FAIL  $*"; ok=1; }
good() { echo "ok    $*"; }
P=http://127.0.0.1:3128
for u in arena-p1 arena-p2 arena-codex; do
  for url in https://api.anthropic.com https://api.openai.com https://api.finic.ai https://github.com https://pypi.org; do
    if as $u curl -s -o /dev/null -m 5 "$url"; then bad "$u reaches $url directly"; else good "$u can't reach $url directly"; fi
  done
  for url in https://api.anthropic.com https://api.openai.com; do
    if as $u curl -s -o /dev/null -m 10 -x $P "$url"; then good "$u reaches $url through the proxy"; else bad "$u can't reach $url through the proxy"; fi
  done
  for url in https://api.finic.ai https://github.com https://pypi.org; do
    if as $u curl -s -o /dev/null -m 10 -x $P "$url"; then bad "the proxy lets $u reach $url"; else good "the proxy blocks $url for $u"; fi
  done
  if as $u ls /srv/arena >/dev/null 2>&1; then bad "$u can list /srv/arena"; else good "$u can't list /srv/arena"; fi
  if as $u test -r /home/ubuntu; then bad "$u can read /home/ubuntu"; else good "$u can't read /home/ubuntu"; fi
  if [ $u != arena-codex ] && as $u sh -c 'ls -a ~/.codex ~/.claude 2>/dev/null | grep -qE "auth.json|credentials"'; then
    bad "$u has a login file"; fi
  for d in /srv/arena/*/; do
    [ -d "$d" ] || continue
    g=$(stat -c %G "$d")
    if [ "$g" = "$u" ]; then continue; fi
    if as $u ls "$d" >/dev/null 2>&1; then bad "$u can read $d"; else good "$u can't read $d"; fi
  done
done
if pgrep -a -u arena-p1,arena-p2,arena-codex 2>/dev/null | grep -qE "API_KEY=|OAUTH_TOKEN=|GATEWAY_KEY=|GRAYSON_TOKEN="; then
  bad "a credential is on a player's command line"; else good "no credential on a player's command line"; fi
if sudo -n -l 2>/dev/null | grep -q "NOPASSWD: /usr/bin/env"; then bad "the harness may still sudo env"; fi
exit $ok
