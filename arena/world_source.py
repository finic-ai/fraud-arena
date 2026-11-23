"""Fraud World as Fraud Arena plays it: the version published on Runmere's community hub, in one tier per board.

fraud-world.json pins the hub version (world id, version number, and the SHA-256 of its world.yaml) and each tier's
parameters. The world's files come with the Runmere engine ($FA_RUNMERE, default /opt/rm-main): its
worlds/fraud-world/world.yaml must be the pinned file, so every run plays the world anyone can read on the hub.

    python -m arena.world_source check     # the pin matches the hub, and the engine's world matches the pin
    python -m arena.world_source path arena

A tier is the world with some parameters changed (`arena`: no independent fraudsters, all fraud is the rings';
`tasks`: the world's own rates). Tier copies are written to a cache folder next to links to the engine's pools and
plugins, so the world's relative imports still resolve.
"""

import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PIN = json.loads((REPO / "fraud-world.json").read_text())
CACHE = Path(os.environ.get("FA_WORLD_CACHE") or Path.home() / ".cache" / "fraud-arena" / "worlds")


def runmere_root():
    root = Path(os.environ.get("FA_RUNMERE") or "/opt/rm-main")
    if not (root / PIN["entry"]).is_file():
        sys.exit(f"no Fraud World at {root / PIN['entry']}: set FA_RUNMERE to the Runmere engine's folder")
    return root


def source():
    """The engine's world.yaml, checked against the pinned hub version."""
    path = runmere_root() / PIN["entry"]
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != PIN["sha256"]:
        sys.exit(f"{path} is not Fraud World v{PIN['version']} as published on the hub ({PIN['hub']}):\n"
                 f"  sha256 {got}, pinned {PIN['sha256']}. Use the engine release that ships it.")
    return path


def set_param(text, name, value):
    """The world text with one parameter's value changed (block or one-line form)."""
    pat = re.compile(rf"(\n    {re.escape(name)}:(?:\n      value:| {{value:) )([-+0-9.eE]+)")
    out, n = pat.subn(lambda m: m.group(1) + repr(float(value)), text)
    if n != 1:
        raise ValueError(f"parameter `{name}`: found {n} places to set it, not 1")
    return out


def tier(name):
    """The path of the world for one tier (`arena` or `tasks`), written once per pinned version."""
    params = PIN["tiers"][name]["params"]
    src = source()
    root = runmere_root()
    base = CACHE / PIN["sha256"][:16]
    folder = base / Path(PIN["entry"]).parent
    path = folder / f"world.{name}.yaml"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        # Everything the world reads by relative path: its own folder's files, and the engine's plugin registry.
        for item in src.parent.iterdir():
            link = folder / item.name
            if not item.name.startswith("world.") and not link.exists():
                link.symlink_to(item)
        if not (base / "registry").exists():
            (base / "registry").symlink_to(root / "registry")
        text = src.read_text()
        for k, v in params.items():
            text = set_param(text, k, v)
        lock = src.with_suffix(".lock")
        if lock.exists():
            path.with_suffix(".lock").write_text(lock.read_text())
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(path)
    return path


def hub_text():
    url = PIN["api"].format(world=PIN["world_id"], version=PIN["version"])
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["text"]


def check():
    hub = hashlib.sha256(hub_text().encode()).hexdigest()
    ok = hub == PIN["sha256"]
    print(f"hub {PIN['hub']} v{PIN['version']}: sha256 {hub[:16]} {'matches' if ok else 'DIFFERS FROM'} the pin")
    src = source()
    print(f"engine world {src}: matches the pin")
    for name in PIN["tiers"]:
        print(f"tier {name}: {tier(name)}")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1:] or ["check"]
    if cmd[0] == "check":
        sys.exit(0 if check() else 1)
    if cmd[0] == "path":
        print(tier(cmd[1]))
