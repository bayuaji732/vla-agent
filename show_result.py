"""
Usage:
    uv run python show_result.py              # latest trajectory
    uv run python show_result.py <filename>   # specific file
"""
import json
import sys
from pathlib import Path

# ── Load trajectory ───────────────────────────────────────────────────────────
trajectories_dir = Path("trajectories")

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
else:
    files = sorted(trajectories_dir.glob("*.json"), reverse=True)
    if not files:
        print("No trajectories found. Run the agent first.")
        sys.exit(1)
    path = files[0]

data = json.loads(path.read_text(encoding="utf-8"))

# ── Helpers ───────────────────────────────────────────────────────────────────
W = 68

def bar(char="─"): print(char * W)
def section(title): bar(); print(f"  {title}"); bar()

def wrap(text, indent=0):
    """Print text with word-wrap at W chars, preserving newlines."""
    prefix = " " * indent
    for paragraph in str(text).split("\n"):
        words = paragraph.split()
        if not words:
            print()
            continue
        line_words, line_len = [], indent
        for w in words:
            if line_len + len(w) + 1 > W:
                print(prefix + " ".join(line_words))
                line_words, line_len = [w], indent + len(w)
            else:
                line_words.append(w)
                line_len += len(w) + 1
        if line_words:
            print(prefix + " ".join(line_words))

# ── Header ────────────────────────────────────────────────────────────────────
bar("═")
print(f"  TASK   : {data['task']}")
print(f"  STATUS : {data['status'].upper()}  |  STEPS: {len(data['steps'])}")
bar("═")

# ── Answer block ──────────────────────────────────────────────────────────────
# Prefer explicit final_answer, then last "done" rationale
answer = data.get("final_answer") or ""
if not answer:
    done_rationales = [
        s["action"].get("rationale", "")
        for s in data["steps"]
        if s["action"]["type"] == "done" and s["action"].get("rationale", "").strip()
    ]
    if done_rationales:
        answer = done_rationales[-1]

if answer:
    print()
    section("ANSWER")
    print()
    wrap(answer, indent=2)
    print()
    bar("═")

# ── Plan ──────────────────────────────────────────────────────────────────────
if data.get("plan"):
    print()
    print(f"  PLAN  ({len(data['plan']['steps'])} steps)")
    bar()
    for s in data["plan"]["steps"]:
        print(f"  {s['index']:>2}.  {s['description']}")
    print()

# ── Execution log ─────────────────────────────────────────────────────────────
print()
print(f"  EXECUTION LOG")
bar()

for i, step in enumerate(data["steps"], 1):
    status    = step["step"]["status"]
    desc      = step["step"]["description"]
    action    = step["action"]["type"].upper()
    rationale = (step["action"].get("rationale") or "").strip()
    retries   = step["step"].get("retries", 0)
    icon      = "✓" if step["success"] else "✗"

    print()
    print(f"  {icon}  Step {i}  [{status}]")
    wrap(desc, indent=6)
    print(f"      action : {action}")
    if rationale:
        print(f"      result : ", end="")
        lines = rationale.split("\n")
        print(lines[0])
        for l in lines[1:]:
            wrap(l, indent=15)
    if retries:
        print(f"      retries: {retries}")
    if step.get("reflection"):
        r = step["reflection"]
        print(f"      reflect: ", end="")
        wrap(r, indent=15)
    if step.get("error"):
        print(f"      error  : {step['error'][:100]}")

print()
bar("═")
print(f"  FILE: {path}")
bar("═")