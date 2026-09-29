"""Rebuild the scratch code lost with /tmp from the session transcripts: replay Write / Edit tool calls and
`cat > f <<'EOF'` / `python3 - <<'EOF'` patchers whose target is under a scratchpad dir, in transcript order.
    python3 replay.py <dry|run>
Everything is remapped from /tmp/.../scratchpad/... to ~/work/... . Analysis heredocs (pixi run python) are skipped.
"""
import json, os, re, subprocess, sys
from pathlib import Path

DRY = sys.argv[1] != "run"
NEW = str(Path.home() / "work")
PROJ = Path.home() / ".claude/projects/-home-gchurchill-src-reblock"
SESSIONS = [("ab8d38f1-74d4-4fea-83ad-93f582da4739", {"harness.py", "study220.py", "flagship.py"}),
            ("fe8870c0-fbd9-4712-ac98-aebcb951b199", None)]
PAT = re.compile(r"/tmp/claude-1641171234/-home-gchurchill-src-reblock/[0-9a-f-]{36}/scratchpad")


def remap(s): return PAT.sub(NEW, s)


def under_scratch(p): return p.startswith(NEW + "/")


def heredocs(cmd):
    lines, out, cwd, i = cmd.split("\n"), [], None, 0
    svar = {m.group(1): m.group(2) for m in re.finditer(r"(?:^|[\s;&])([A-Z])=(/\S+?)(?=$|[\s;&])", cmd)}
    while i < len(lines):
        ln = lines[i]
        for m in re.finditer(r"cd\s+(\S+)", ln):
            cwd = m.group(1).strip("'\";&")
        if ln.lstrip().startswith("sed -i ") and cwd:
            out.append((ln.strip(), None, cwd, svar))
        if "<<'EOF'" in ln:
            body, j = [], i + 1
            while j < len(lines) and lines[j] != "EOF":
                body.append(lines[j]); j += 1
            out.append((ln, "\n".join(body) + "\n", cwd, svar))
            i = j
        i += 1
    return out


events = []
for sid, allow in SESSIONS:
    for line in open(PROJ / f"{sid}.jsonl"):
        try: rec = json.loads(line)
        except Exception: continue
        c = (rec.get("message") or {}).get("content")
        if not isinstance(c, list): continue
        for b in c:
            if not (isinstance(b, dict) and b.get("type") == "tool_use"): continue
            inp, name = b.get("input", {}), b["name"]
            if name in ("Write", "Edit"):
                fp = remap(inp.get("file_path", ""))
                if under_scratch(fp) and (allow is None or os.path.basename(fp) in allow):
                    events.append((sid[:8], name, fp, inp))
            elif name == "Bash":
                cmd = remap(inp.get("command", ""))
                for hdr, body, cwd, svar in heredocs(cmd):
                    if body is None:
                        if under_scratch(cwd) and re.search(r"\s\S+\.py\s*$", hdr):
                            events.append((sid[:8], "sed", cwd, {"line": hdr}))
                        continue
                    m = re.search(r"cat\s*(>>?)\s*(\S+)\s*<<", hdr)
                    if m:
                        p = m.group(2).strip("'\"")
                        for k, v in svar.items(): p = p.replace("$" + k, v).replace("${" + k + "}", v)
                        if not p.startswith("/") and cwd: p = os.path.join(cwd, p)
                        if under_scratch(p) and (allow is None or os.path.basename(p) in allow):
                            events.append((sid[:8], "cat" + m.group(1), p, {"content": body}))
                    elif re.search(r"python3\s+-\s*<<", hdr) and re.search(r"open\(|write_text", body) \
                            and re.search(r"'w'|\"w\"|write_text", body) \
                            and ((cwd and under_scratch(cwd)) or (NEW + "/" in body and "docs/" not in body
                                                  and "src/reblock" not in body and "memory" not in body)):
                        events.append((sid[:8], "patch", cwd if cwd and under_scratch(cwd) else NEW, {"body": remap(body)}))

print(len(events), "events")
for sid, kind, path, inp in events:
    print(f"  {sid} {kind:6s} {path.replace(NEW, '~/work')}")
if DRY: sys.exit()
bad = 0
for sid, kind, path, inp in events:
    if kind == "Write" or kind == "cat>":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(remap(inp.get("content", "")))
    elif kind == "cat>>":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f: f.write(remap(inp["content"]))
    elif kind == "Edit":
        s = Path(path).read_text(); old, new = remap(inp["old_string"]), remap(inp["new_string"])
        if old not in s: print("EDIT MISS", path); bad += 1; continue
        Path(path).write_text(s.replace(old, new) if inp.get("replace_all") else s.replace(old, new, 1))
    elif kind == "sed":
        r = subprocess.run(inp["line"], shell=True, cwd=path, capture_output=True, text=True)
        if r.returncode: print("SED FAIL", inp["line"][:80], r.stderr[-200:]); bad += 1
    elif kind == "patch":
        r = subprocess.run(["python3", "-"], input=inp["body"], text=True, cwd=path, capture_output=True)
        if r.returncode: print("PATCH FAIL in", path, r.stderr[-300:]); bad += 1
print("done;", bad, "problems")
