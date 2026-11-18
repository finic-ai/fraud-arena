#!/bin/bash
# Turns an EC2 work box into a pilot sandbox host. The harness runs as `ubuntu`; each player's sessions run as that
# player's own user, which can read only its own folder and reach only loopback. Run as root, then start a new login
# session for `ubuntu` so the harness has its new groups. Idempotent.
set -euo pipefail
# /srv/arena holds a private folder per player (fraud_arena/sandbox.py): its workspace, its tables and its deployed
# bundles, readable only by that player's user and the harness. Players can't list /srv/arena. Episode state
# (checkpoints, scoring exports, the gateway's usage log) stays under /home/ubuntu, which no player can read.
chmod 750 /home/ubuntu
# A folder left by the shared-group pilot is readable by its group: keep it, but out of every player's reach.
if [ -d /srv/arena ] && [ "$(stat -c %a /srv/arena)" != 711 ]; then
  mv /srv/arena "/srv/arena.shared-$(date +%Y%m%d%H%M%S)" && chmod 700 /srv/arena.shared-*
fi
install -d -o ubuntu -g ubuntu -m 711 /srv/arena

# Players reach the network only on loopback. With `--credentials api-key` their one route to a model is the
# harness's gateway, which holds the keys. With subscriptions they reach the providers through this proxy: CONNECT
# only, and only to these hosts.
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tinyproxy >/dev/null
cat > /etc/tinyproxy/allow <<'EOF2'
^api\.anthropic\.com$
^(console|claude|platform)\.anthropic\.com$
^statsig\.anthropic\.com$
^claude\.ai$
^api\.openai\.com$
^chatgpt\.com$
^(auth|ab)\.openai\.com$
EOF2
cat > /etc/tinyproxy/tinyproxy.conf <<'EOF2'
User tinyproxy
Group tinyproxy
Port 3128
Listen 127.0.0.1
Timeout 600
LogLevel Connect
MaxClients 200
Filter "/etc/tinyproxy/allow"
FilterType ere
FilterDefaultDeny Yes
FilterURLs Off
ConnectPort 443
EOF2
systemctl enable --now tinyproxy >/dev/null 2>&1 && systemctl restart tinyproxy

# One user per player (arena-p1..arena-p12). Each user's own group holds only it and the harness user, so no player
# can read another player's folder: an attacker's workspace holds its ring's view, which is truth. With subscriptions,
# every Codex session runs as arena-codex with its one ChatGPT login (copies of a login invalidate each other when one
# refreshes), so Codex players share that user, and the harness refuses Codex on both sides of a game.
userdel -r agent >/dev/null 2>&1 || true
getent group arena >/dev/null && groupdel arena
for u in $(seq -f 'arena-p%g' 1 12) arena-codex; do
  id $u >/dev/null 2>&1 || useradd --create-home --shell /bin/bash --user-group $u
  [ "$(id -gn $u)" = "$u" ] || { getent group $u >/dev/null || groupadd $u; usermod -g $u $u; }
  chmod 700 /home/$u
  sed -i '/umask 002/d' /home/$u/.profile 2>/dev/null || true
  usermod -aG $u ubuntu
  uid=$(id -u $u)
  for t in iptables ip6tables; do
    $t -C OUTPUT -m owner --uid-owner $uid -o lo -j ACCEPT 2>/dev/null || $t -A OUTPUT -m owner --uid-owner $uid -o lo -j ACCEPT
    $t -C OUTPUT -m owner --uid-owner $uid -j REJECT 2>/dev/null || $t -A OUTPUT -m owner --uid-owner $uid -j REJECT
  done
done
install -d -o arena-codex -g arena-codex -m 700 /home/arena-codex/.codex
aws secretsmanager get-secret-value --region us-west-2 --secret-id fraud-arena/codex-auth --query SecretString \
  --output text > /home/arena-codex/.codex/auth.json
chown arena-codex:arena-codex /home/arena-codex/.codex/auth.json && chmod 600 /home/arena-codex/.codex/auth.json

# Agent CLIs, system-wide so the players can run them.
command -v claude >/dev/null || npm install -g --silent @anthropic-ai/claude-code
command -v codex >/dev/null || npm install -g --silent @openai/codex

# The harness starts a player's session through arena-exec, which reads the session's environment from a file in
# the player's folder, so no secret is ever on a command line. The harness user may run that, as a player, and
# nothing else.
install -o root -g root -m 755 "$(dirname "$0")/arena-exec" /usr/local/bin/arena-exec
{ echo "Runas_Alias ARENA = arena-codex, $(seq -s, -f 'arena-p%g' 1 12)"; echo "ubuntu ALL=(ARENA) NOPASSWD: /usr/local/bin/arena-exec"; } > /etc/sudoers.d/fraud-arena
chmod 440 /etc/sudoers.d/fraud-arena && visudo -cf /etc/sudoers.d/fraud-arena >/dev/null

echo "sandbox ready: players arena-p1..arena-p12 and arena-codex, proxy http://127.0.0.1:3128"
