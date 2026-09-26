# Phase X of the Spark rebuild, on the hub node: build the files several nodes contributed to (the attention file and the two
# range-boundary shards) from every node's PIECES plus the served file. Env: PLAN, OURS, OUT, PIECES. Exits non-zero if a
# contributing node's pieces are missing or any quantized tensor of a shared file was not produced.
import glob, json, os, sys
from collections import defaultdict
from safetensors import safe_open
from safetensors.torch import load_file, save_file

E = os.environ; OURS, OUT, PIECES = E["OURS"], E["OUT"], E["PIECES"]
sidx = json.load(open(f"{OURS}/model.safetensors.index.json"))["weight_map"]
owners = defaultdict(set)
for n, v in json.load(open(E["PLAN"]))["nodes"].items():
    for g in v["groups"]:
        for m in g: owners[sidx[m]].add(n)
bad = 0
for f in sorted(x for x, o in owners.items() if len(o) > 1):
    parts = sorted(p for p in glob.glob(f"{PIECES}/{f}.*") if not p.endswith(".tmp")); have = {p.rsplit(".", 1)[1] for p in parts}
    if have != owners[f]: print(f"FAIL {f}: pieces from {sorted(have)}, expected {sorted(owners[f])}"); bad += 1; continue
    got = {}
    for p in parts: got.update(load_file(p))
    with safe_open(f"{OURS}/{f}", "pt", device="cpu") as fs:
        qn = [k for k in fs.keys() if k.endswith(".weight") and k[:-7] + ".weight_scale" in sidx]
        miss = [k for k in qn if k not in got]
        if miss: print(f"FAIL {f}: {len(miss)} quantized tensors missing, e.g. {miss[:2]}"); bad += 1; continue
        out = {k: (got[k] if k in got else fs.get_tensor(k)) for k in fs.keys()}
        save_file(out, f"{OUT}/{f}.tmp", metadata=fs.metadata()); os.replace(f"{OUT}/{f}.tmp", f"{OUT}/{f}")
    print(f"FINAL {f}: {len(qn)} quantized tensors from {sorted(have)}, {len(out) - 3 * len(qn)} copied", flush=True)
print("REBUILD_SHARED", "PASS" if not bad else f"FAIL ({bad})"); sys.exit(1 if bad else 0)
