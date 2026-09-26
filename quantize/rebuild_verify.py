# Verify a rebuilt checkpoint against the served one before anything boots it: every served file present, identical tensor
# names, dtypes and shapes, every quantized weight actually changed, every other tensor byte-identical (checked on a sample),
# every fused group sharing one global scale, and the non-weight files present.
# Usage: python3 rebuild_verify.py <served dir> <rebuilt dir> [ONLY_FILES comma list, tests]
import hashlib, json, os, random, sys
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuild_core import groups

ours, new = sys.argv[1], sys.argv[2]; only = set(filter(None, (sys.argv[3] if len(sys.argv) > 3 else "").split(",")))
sidx = json.load(open(f"{ours}/model.safetensors.index.json"))["weight_map"]
files = sorted(f for f in set(sidx.values()) if not only or f in only)
Q = [n for n in sidx if n.endswith(".weight") and n[:-7] + ".weight_scale" in sidx and sidx[n] in files]
bad, s2 = [], {}
h = lambda t: hashlib.sha256(t.contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes()).hexdigest()
rng = random.Random(0)
for f in files:
    if not os.path.exists(f"{new}/{f}"): bad.append(f"missing {f}"); continue
    with safe_open(f"{ours}/{f}", "pt", device="cpu") as a, safe_open(f"{new}/{f}", "pt", device="cpu") as b:
        ka, kb = set(a.keys()), set(b.keys())
        if ka != kb: bad.append(f"{f}: tensor names differ ({len(ka ^ kb)})"); continue
        for k in ka:
            sa, sb = a.get_slice(k), b.get_slice(k)
            if (sa.get_dtype(), sa.get_shape()) != (sb.get_dtype(), sb.get_shape()): bad.append(f"{k}: {sb.get_dtype()} {sb.get_shape()} vs served {sa.get_dtype()} {sa.get_shape()}")
            if k.endswith(".weight_scale_2"): s2[k] = b.get_tensor(k).item()
        qset = {k for k in ka if k.endswith(".weight") and k[:-7] + ".weight_scale" in sidx}
        for k in rng.sample(sorted(qset), min(3, len(qset))):
            if h(a.get_tensor(k)) == h(b.get_tensor(k)): bad.append(f"{k}: quantized weight unchanged from served")
        rest = sorted(k for k in ka if k not in qset and not k.endswith((".weight_scale", ".weight_scale_2")))
        for k in rng.sample(rest, min(3, len(rest))):
            if h(a.get_tensor(k)) != h(b.get_tensor(k)): bad.append(f"{k}: non-quantized tensor differs from served")
unshared = [g for g in groups(Q) if len(g) > 1 and len({s2.get(m[:-7] + ".weight_scale_2") for m in g}) != 1]
bad += [f"fused group does not share one global scale: {g[0]} ..." for g in unshared]
if not only:
    for x in sorted(os.listdir(ours)):
        if not x.endswith(".safetensors") and not os.path.exists(f"{new}/{x}"): bad.append(f"missing non-weight file {x}")
print(f"checked {len(files)} files, {len(Q)} quantized weights, {len([g for g in groups(Q) if len(g) > 1])} fused groups")
for b_ in bad[:20]: print("FAIL", b_)
print("REBUILD_VERIFY", "PASS" if not bad else f"FAIL ({len(bad)})"); sys.exit(1 if bad else 0)
