"""Each player's files are private to it: a folder per player that only its Unix user and the harness can read, and
sessions started as the player with their environment in a file in that folder, never on a command line, where
`ps` shows it to every user of the box. See sandbox/setup-box.sh.
"""
from __future__ import annotations

import grp
import json
import os
import pwd
import shutil
from pathlib import Path

EXEC = "/usr/local/bin/arena-exec"


def private_dir(path: Path, user: str | None) -> Path:
    """`path`, readable and writable only by `user` (through its own group, which the harness user is in) and the
    harness. Setgid, so what the harness writes inside keeps the player's group."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if user:
        group = grp.getgrgid(pwd.getpwnam(user).pw_gid).gr_name
        shutil.chown(path, group=group)
        os.chmod(path, 0o2770)
    return path


def as_user(user: str, env: dict[str, str], argv: list[str], env_file: Path) -> list[str]:
    """`argv` run as `user` with exactly `env`, which goes through `env_file` in the player's folder."""
    env_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = env_file.with_name(env_file.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
    with os.fdopen(fd, "w") as f:
        json.dump({k: v for k, v in env.items() if k not in ("HOME", "USER", "LOGNAME")}, f)
    os.replace(tmp, env_file)
    return ["sudo", "-n", "-u", user, "-H", EXEC, str(env_file), *argv]
