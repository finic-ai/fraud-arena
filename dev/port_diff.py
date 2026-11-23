"""Re-apply a commit's world.yaml changes onto another version of the file, group by group.

Each group of removed/added lines in the commit's diff is applied where the removed text occurs exactly once in the
target; groups whose removed text isn't found (the target changed there) are reported for manual porting. Pure
insertions are placed after their preceding context line when that line occurs exactly once."""
import subprocess, sys

repo, commit, path, target = sys.argv[1:5]
diff = subprocess.run(["git", "-C", repo, "show", "--format=", "-U1", commit, "--", path], capture_output=True, text=True, check=True).stdout
text = open(target).read()
groups, cur = [], None
for line in diff.splitlines():
    if line.startswith(("diff ", "index ", "--- ", "+++ ", "@@")):
        cur = None
        if line.startswith("@@"):
            groups.append({"ctx": None, "rem": [], "add": []}); cur = groups[-1]
        continue
    if cur is None:
        continue
    if line.startswith("-"):
        if cur["add"]:  # a new change group after an addition
            groups.append({"ctx": None, "rem": [], "add": []}); cur = groups[-1]
        cur["rem"].append(line[1:])
    elif line.startswith("+"):
        cur["add"].append(line[1:])
    else:
        if cur["rem"] or cur["add"]:
            groups.append({"ctx": line[1:], "rem": [], "add": []}); cur = groups[-1]
        else:
            cur["ctx"] = line[1:]
applied, missed = 0, []
for g in groups:
    if not g["rem"] and not g["add"]:
        continue
    if g["rem"]:
        old = "\n".join(g["rem"]) + "\n"
        n = text.count(old)
        if n == 1:
            text = text.replace(old, "\n".join(g["add"]) + "\n" if g["add"] else ""); applied += 1
        else:
            missed.append((n, g))
    else:  # pure insertion after context
        ctx = (g["ctx"] or "") + "\n"
        n = text.count(ctx) if g["ctx"] else 0
        if n == 1:
            text = text.replace(ctx, ctx + "\n".join(g["add"]) + "\n"); applied += 1
        else:
            missed.append((n, g))
open(target, "w").write(text)
print(f"applied {applied} groups, missed {len(missed)}")
for n, g in missed:
    first = (g["rem"] or g["add"])[0].strip()
    print(f"  MISSED (found {n}x): ctx={ (g['ctx'] or '').strip()[:90]!r} first={first[:150]!r}")
