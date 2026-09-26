# Phase Q driver: run rebuild_node.py in bounded chunks so no single process lives long enough to accumulate memory
# (2026-09-24 02:50: one process per node grew ~100 GB in 16 min and oomwrap killed it; the cause is not in quantize_group).
# A chunk is a set of files linked by this node's remaining fused groups (a group never splits across chunks), about
# CHUNK_FILES files each, run in a fresh process. Resumable: groups whose files are already final are skipped.
# Env: everything rebuild_node.py takes, plus CHUNK_FILES (4) and TOTAL_MAX_MIN (stop between chunks once exceeded).
# Prints REBUILD_CHUNKS PASS when every chunk passes.
import json, os, subprocess, sys, time

E = os.environ; NODE, OUT, CH = E["NODE"], E["OUT"], int(E.get("CHUNK_FILES", "4"))
plan = json.load(open(E["PLAN"]))["nodes"][NODE]
sidx = json.load(open(f"{E['OURS']}/model.safetensors.index.json"))["weight_map"]
bidx = json.load(open(f"{E['BF16']}/model.safetensors.index.json"))["weight_map"]
groups = [g for g in plan["groups"] if not all(os.path.exists(f"{OUT}/{sidx[m]}") for m in g)]
if E.get("SKIP_MISSING") == "1":  # tests only
    groups = [g for g in groups if all(os.path.exists(f"{E['BF16']}/{bidx[m]}") for m in g)]
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for g in groups:
    fs = [sidx[m] for m in g]
    for f in fs: find(f)
    for f in fs[1:]: parent[find(f)] = find(fs[0])
comps = {}
for f in list(parent): comps.setdefault(find(f), []).append(f)
chunks, cur = [], []
for c in sorted((sorted(c) for c in comps.values()), key=lambda c: c[0]):
    if cur and len(cur) + len(c) > CH: chunks.append(cur); cur = []
    cur += c
if cur: chunks.append(cur)
mem = lambda: next((int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable")), -1) if os.path.exists("/proc/meminfo") else -1
budget, t0 = float(E.get("TOTAL_MAX_MIN", "0")), time.time()
print(f"[{time.strftime('%T')}] {NODE}: {len(groups)} groups left in {len(chunks)} chunks", flush=True)
node = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rebuild_node.py")
for i, ch in enumerate(chunks):
    if budget and (time.time() - t0) / 60 > budget: print(f"REBUILD_CHUNKS FAIL: over the {budget:.0f} min budget before chunk {i + 1}"); sys.exit(4)
    rc = subprocess.call([sys.executable, node], env=dict(E, ONLY_FILES=",".join(ch), SKIP_FINAL="1"))
    print(f"[{time.strftime('%T')}] {NODE} chunk {i + 1}/{len(chunks)} ({ch[0]} .. {ch[-1]}): exit {rc}, host MemAvailable {mem()} MiB", flush=True)
    if rc != 0: print("REBUILD_CHUNKS FAIL"); sys.exit(rc)
print(f"REBUILD_CHUNKS PASS in {(time.time() - t0) / 60:.1f} min", flush=True)
