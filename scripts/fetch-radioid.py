#!/usr/bin/python3 -I
"""Select and download the recordings for the radio identification cases (130-134).

usage: scripts/fetch-radioid.py [data/rtl433id]

Source: merbanan/rtl_433_tests on GitHub. Writes index.txt (device file), index-sessions.txt
(protocol session file) and the .cu8 files (about 390 MB). Selection rules are in TECHNICAL.md 22.1.
The random choice of at most 24 (or 12) recordings per folder is seeded, but the result depends on the
repository's contents on the day. Treat the downloads as untrusted data.
Written from the commands used in the session; this script itself has not been run end to end."""
import collections, json, os, random, subprocess, sys, urllib.request

out = sys.argv[1] if len(sys.argv) > 1 else "data/rtl433id"
os.makedirs(out, exist_ok=True)
RAW = "https://raw.githubusercontent.com/merbanan/rtl_433_tests/master/"
tree = json.load(urllib.request.urlopen("https://api.github.com/repos/merbanan/rtl_433_tests/git/trees/master?recursive=1"))["tree"]
folders = collections.defaultdict(list)
for t in tree:
    path, size = t["path"], t.get("size", 0)
    q = path.split("/")
    b = q[-1]
    if not (path.startswith("tests/") and b.endswith(".cu8") and len(q) >= 4):
        continue
    if not ("_250k" in b or b.startswith("gfile") or b.startswith("g0")) or "1024k" in b or "1000k" in b:
        continue
    if 60000 <= size <= 4000000:
        folders[(q[1], q[2])].append(path)
want = {}

def name(path):
    return path[6:].replace("/", "_")

# Same-session set: the largest folder of each family with 10 or more recordings, at most 24 each.
random.seed(1)
best = {}
for (fam, sub), v in sorted(folders.items(), key=lambda kv: -len(kv[1])):
    if len(v) >= 10 and fam not in best:
        best[fam] = (sub, v)
with open(os.path.join(out, "index.txt"), "w") as ix:
    for fam, (sub, v) in sorted(best.items()):
        v = sorted(v); random.shuffle(v)
        for path in v[:24]:
            ix.write("%s_%s %s\n" % (fam, sub, name(path))); want[name(path)] = path
# Cross-session set: families with two or more numbered folders of 5 or more recordings.
random.seed(2)
fams = collections.defaultdict(list)
for (fam, sub), v in folders.items():
    if len(v) >= 5:
        fams[fam].append((sub, v))
with open(os.path.join(out, "index-sessions.txt"), "w") as ix:
    for fam, subs in sorted(fams.items()):
        subs = sorted(subs)[:4]
        if len(subs) < 2 or not all(s.isdigit() for s, _ in subs):
            continue
        for si, (sub, v) in enumerate(subs):
            v = sorted(v); random.shuffle(v)
            for path in v[:12]:
                ix.write("%s %d %s\n" % (fam, si, name(path))); want[name(path)] = path
cfg = os.path.join(out, "curl.cfg")
with open(cfg, "w") as c:
    for n, path in sorted(want.items()):
        if not os.path.exists(os.path.join(out, n)):
            c.write('url = "%s%s"\noutput = "%s"\n' % (RAW, path.replace("#", "%23"), n))
print(len(want), "recordings selected; downloading the missing ones")
subprocess.run(["curl", "-sS", "--fail", "-L", "--parallel", "--parallel-max", "12", "--retry", "2", "-K", "curl.cfg"], cwd=out)
