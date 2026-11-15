#!/bin/bash
# Turns an EC2 work box into a pilot sandbox host. The harness runs as `ubuntu`; each agent session runs as the
# `agent` user, which can read only its episode workspace and its institution's tables, and reaches the network only
# through a local proxy that allows the model providers and Grayson. Run as root. Idempotent.
set -euo pipefail
R=us-west-2

id agent >/dev/null 2>&1 || useradd --create-home --shell /bin/bash agent
chmod 750 /home/ubuntu
install -d -o agent -g agent -m 700 /home/agent/.codex /home/agent/.claude

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
^api\.finic\.ai$
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

# The harness user may start sessions as `agent` without a password, and nothing else.
echo "ubuntu ALL=(agent) NOPASSWD: /usr/bin/env" > /etc/sudoers.d/fraud-arena && chmod 440 /etc/sudoers.d/fraud-arena

echo "sandbox ready: agent uid $U, proxy http://127.0.0.1:3128"
