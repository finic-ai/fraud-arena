#!/bin/bash
# Turns an EC2 work box into a pilot sandbox host. The harness runs as `ubuntu`; each agent session runs as the
# `agent` user, which can read only its episode workspace and its institution's tables, and reaches the network only
# through a local proxy that allows the model providers and Grayson. Run as root. Idempotent.
set -euo pipefail
R=us-west-2

id agent >/dev/null 2>&1 || useradd --create-home --shell /bin/bash agent
chmod 750 /home/ubuntu
install -d -o agent -g agent -m 700 /home/agent/.codex /home/agent/.claude
# /srv/arena: workspaces and the tables served to agents; shared by the harness (ubuntu) and `agent`. Episode state
# (checkpoints, scoring exports) stays under /home/ubuntu, which `agent` can't read.
getent group arena >/dev/null || groupadd arena
usermod -aG arena agent && usermod -aG arena ubuntu
install -d -o ubuntu -g arena -m 2770 /srv/arena
grep -q 'umask 002' /home/agent/.profile 2>/dev/null || echo 'umask 002' >> /home/agent/.profile

# Agent CLIs, system-wide so `agent` can run them.
command -v codex >/dev/null || npm install -g --silent @openai/codex
aws secretsmanager get-secret-value --region $R --secret-id fraud-arena/codex-auth --query SecretString \
  --output text > /home/agent/.codex/auth.json
chown agent:agent /home/agent/.codex/auth.json && chmod 600 /home/agent/.codex/auth.json

# The proxy: CONNECT only, and only to these hosts.
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tinyproxy >/dev/null
cat > /etc/tinyproxy/allow <<'EOF'
^api\.anthropic\.com$
^(console|claude|platform)\.anthropic\.com$
^statsig\.anthropic\.com$
^claude\.ai$
^api\.openai\.com$
^chatgpt\.com$
^(auth|ab)\.openai\.com$
EOF
cat > /etc/tinyproxy/tinyproxy.conf <<'EOF'
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
EOF
systemctl restart tinyproxy

# `agent` may talk to loopback (the proxy, its own MCP tool server) and nothing else.
U=$(id -u agent)
iptables -C OUTPUT -m owner --uid-owner "$U" -o lo -j ACCEPT 2>/dev/null || iptables -A OUTPUT -m owner --uid-owner "$U" -o lo -j ACCEPT
iptables -C OUTPUT -m owner --uid-owner "$U" -j REJECT 2>/dev/null || iptables -A OUTPUT -m owner --uid-owner "$U" -j REJECT
ip6tables -C OUTPUT -m owner --uid-owner "$U" -o lo -j ACCEPT 2>/dev/null || ip6tables -A OUTPUT -m owner --uid-owner "$U" -o lo -j ACCEPT
ip6tables -C OUTPUT -m owner --uid-owner "$U" -j REJECT 2>/dev/null || ip6tables -A OUTPUT -m owner --uid-owner "$U" -j REJECT

# One user per Claude Code player (arena-p1..arena-p12), each with its own home and workspace. All Codex players run
# as one user, arena-codex, with one ChatGPT login: copies of a login invalidate each other when one refreshes, and a
# login home can't be shared across users (Codex writes its files private). Codex players of a game are then one
# team that can see each other's workspaces (fine within a side; self-play needs a second login).
for u in $(seq -f 'arena-p%g' 1 12) arena-codex; do
  id $u >/dev/null 2>&1 || useradd --create-home --shell /bin/bash $(getent group $u >/dev/null && echo -g $u) $u
  usermod -aG arena $u
  grep -q 'umask 002' /home/$u/.profile 2>/dev/null || echo 'umask 002' >> /home/$u/.profile
  uid=$(id -u $u)
  for t in iptables ip6tables; do
    $t -C OUTPUT -m owner --uid-owner $uid -o lo -j ACCEPT 2>/dev/null || $t -A OUTPUT -m owner --uid-owner $uid -o lo -j ACCEPT
    $t -C OUTPUT -m owner --uid-owner $uid -j REJECT 2>/dev/null || $t -A OUTPUT -m owner --uid-owner $uid -j REJECT
  done
done
install -d -o arena-codex -g arena-codex -m 700 /home/arena-codex/.codex
aws secretsmanager get-secret-value --region $R --secret-id fraud-arena/codex-auth --query SecretString \
  --output text > /home/arena-codex/.codex/auth.json
chown arena-codex:arena-codex /home/arena-codex/.codex/auth.json && chmod 600 /home/arena-codex/.codex/auth.json

# The harness user may start sessions as the players without a password, and nothing else.
{ echo "Runas_Alias ARENA = agent, arena-codex, $(seq -s, -f 'arena-p%g' 1 12)"; echo "ubuntu ALL=(ARENA) NOPASSWD: /usr/bin/env"; } > /etc/sudoers.d/fraud-arena
chmod 440 /etc/sudoers.d/fraud-arena && visudo -cf /etc/sudoers.d/fraud-arena >/dev/null

echo "sandbox ready: agent uid $U, proxy http://127.0.0.1:3128"
