# Phase Q of the Spark rebuild, one node: quantize the fused groups plan.json gives this node (BF16 -> NVFP4, LLMC MSE e1.0,
# rebuild_core) and write, for every served file those groups touch, either the FINAL file (the served file with its quantized
# tensors replaced, everything else copied byte-for-byte) when this node owns all of that file's quantized tensors, or a PIECES
# file when other nodes contribute too (the attention file and the two range-boundary shards; rebuild_shared.py joins those).
# Every result is checked against the served tensor's dtype/shape and fused global-scale sharing; 1 group in SAMPLE_EVERY is
# checked for error vs BF16 against what we serve. Exits non-zero on any failed check.
# Env: NODE, PLAN, BF16 (dir with shards + model.safetensors.index.json), OURS (served dir), OUT, PIECES, DEVICE (cuda|cpu),
#      SKIP_FINAL (1 = resume: skip groups whose files are final), MAX_MIN (abort after 200 groups if the ETA exceeds it), SAMPLE_EVERY (50), ONLY_FILES (comma list, tests), SKIP_MISSING (1 = tests with partial shards), DELETE_BF16 (1 = delete a BF16 shard once nothing needs it)
import json, os, sys, time, torch
from collections import defaultdict
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuild_core import dequant, quantize_group

E = os.environ; NODE, DEV = E["NODE"], E.get("DEVICE", "cuda"); SAMPLE = int(E.get("SAMPLE_EVERY", "50"))
if DEV == "cpu": torch.set_num_threads(int(E.get("THREADS", "8")))
OURS, BF, OUT, PIECES = E["OURS"], E["BF16"], E["OUT"], E["PIECES"]
plan = json.load(open(E["PLAN"]))["nodes"]
sidx = json.load(open(f"{OURS}/model.safetensors.index.json"))["weight_map"]
bidx = json.load(open(f"{BF}/model.safetensors.index.json"))["weight_map"]
owners = defaultdict(set)
for n, v in plan.items():
    for g in v["groups"]:
        for m in g: owners[sidx[m]].add(n)
only = set(filter(None, E.get("ONLY_FILES", "").split(",")))
mine = sorted((g for g in plan[NODE]["groups"] if not only or sidx[g[0]] in only), key=lambda g: (sidx[g[0]], g[0]))
if E.get("SKIP_FINAL") == "1":  # resume: a final file exists only after an atomic rename, so it is complete
    mine = [g for g in mine if not all(os.path.exists(f"{OUT}/{sidx[m]}") for m in g)]
if E.get("SKIP_MISSING") == "1":  # tests only: keep groups whose BF16 shards are present here
    mine = [g for g in mine if all(os.path.exists(f"{BF}/{bidx[m]}") for m in g)]
os.makedirs(OUT, exist_ok=True); os.makedirs(PIECES, exist_ok=True)
log = lambda *a: print(f"[{time.strftime('%T')}] {NODE}", *a, flush=True)

_h = {}
def handle(path):
    if path not in _h: _h[path] = safe_open(path, "pt", device="cpu")
    return _h[path]
served_meta = lambda name: handle(f"{OURS}/{sidx[name]}").get_slice(name)

results, done_files, sample, t0 = defaultdict(dict), set(), [], time.time()
last_use = {}  # BF16 shard -> index of the last group that reads it (to delete early when allowed)
for i, g in enumerate(mine):
    for m in g: last_use[bidx[m]] = i

def flush(upto_file):
    """Write every file this node has fully produced, i.e. all files ordered before upto_file."""
    for f in sorted(results):
        if f in done_files or (upto_file is not None and f >= upto_file): continue
        if os.path.exists(f"{OUT}/{f}"):  # already final from an earlier run: keep it, drop this run's copy
            done_files.add(f); del results[f]; continue
        with safe_open(f"{OURS}/{f}", "pt", device="cpu") as fs:
            if owners[f] == {NODE}:
                qn = [k for k in fs.keys() if k.endswith(".weight") and k[:-7] + ".weight_scale" in sidx]
                miss = [k for k in qn if k not in results[f]]
                assert not miss, f"{f}: {len(miss)} quantized tensors not produced, e.g. {miss[:2]}"
                out = {k: (results[f][k] if k in results[f] else fs.get_tensor(k)) for k in fs.keys()}
                save_file(out, f"{OUT}/{f}.tmp", metadata=fs.metadata()); os.replace(f"{OUT}/{f}.tmp", f"{OUT}/{f}")
                log(f"FINAL {f}: {len(qn)} quantized tensors replaced, {len(out) - 3 * len(qn)} copied")
            else:
                save_file(results[f], f"{PIECES}/{f}.{NODE}.tmp"); os.replace(f"{PIECES}/{f}.{NODE}.tmp", f"{PIECES}/{f}.{NODE}")
                log(f"PIECES {f}: {len(results[f]) // 3} quantized tensors (file shared with {sorted(owners[f] - {NODE})})")
        done_files.add(f); del results[f]

log(f"start: {len(mine)} groups, {len(plan[NODE]['shards'])} BF16 shards, device {DEV}")
for i, g in enumerate(mine):
    if i and sidx[g[0]] != sidx[mine[i - 1][0]]: flush(sidx[g[0]])
    ws = {m: handle(f"{BF}/{bidx[m]}").get_tensor(m) for m in g}
    out = quantize_group(ws, DEV)
    s2_served = []
    for m in g:
        q, s, s2, miss = out[m]; meta = {sfx: served_meta(m[:-7] + sfx) for sfx in (".weight", ".weight_scale", ".weight_scale_2")}
        s2 = s2.reshape(meta[".weight_scale_2"].get_shape())
        for sfx, t in ((".weight", q), (".weight_scale", s), (".weight_scale_2", s2)):
            assert list(t.shape) == meta[sfx].get_shape() and str(t.dtype).split(".")[-1] == meta[sfx].get_dtype().lower().replace("f8_e4m3", "float8_e4m3fn").replace("u8", "uint8").replace("f32", "float32"), \
                f"{m}{sfx}: layout {t.dtype} {tuple(t.shape)} vs served {meta[sfx].get_dtype()} {meta[sfx].get_shape()}"
        assert miss < 0.1, f"{m}: FP4 grid miss {miss}"
        f = sidx[m]; results[f][m], results[f][m[:-7] + ".weight_scale"], results[f][m[:-7] + ".weight_scale_2"] = q.cpu(), s.cpu(), s2.cpu()
        s2_served.append(handle(f"{OURS}/{sidx[m]}").get_tensor(m[:-7] + ".weight_scale_2").item())
    if len(g) > 1:
        assert len(set(s2_served)) == 1, f"served checkpoint does not share one global scale across {g}"
        assert len({results[sidx[m]][m[:-7] + '.weight_scale_2'].item() for m in g}) == 1, f"rebuilt group {g} does not share one global scale"
    if i % SAMPLE == 0:
        for m in g:
            w = ws[m].float(); fs = handle(f"{OURS}/{sidx[m]}")
            served = dequant(*(fs.get_tensor(m[:-7] + x) for x in (".weight", ".weight_scale", ".weight_scale_2")))
            r = results[sidx[m]]; mine_w = dequant(r[m], r[m[:-7] + ".weight_scale"], r[m[:-7] + ".weight_scale_2"])
            sample.append((((mine_w - w).norm() / w.norm()).item(), ((served - w).norm() / w.norm()).item(), w.numel()))
    if E.get("DELETE_BF16") == "1":
        for shard in {bidx[m] for m in g}:
            if last_use[shard] == i: _h.pop(f"{BF}/{shard}", None); os.remove(f"{BF}/{shard}"); log(f"deleted BF16 {shard}")
    if i == 199 and E.get("MAX_MIN"):  # the window has a time budget: bail out early if this node cannot make it
        eta = (len(mine) - 200) / (200 / (time.time() - t0)) / 60
        if eta > float(E["MAX_MIN"]): log(f"REBUILD_NODE ABORT: ETA {eta:.0f} min > budget {E['MAX_MIN']} min"); sys.exit(4)
    if i % 200 == 199:
        import gc; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    if i % 500 == 0 or i == len(mine) - 1:
        rate = (i + 1) / (time.time() - t0); mem = next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable")) if os.path.exists("/proc/meminfo") else -1
        log(f"{i + 1}/{len(mine)} groups, {rate:.2f} groups/s, ETA {(len(mine) - i - 1) / rate / 60:.1f} min, host MemAvailable {mem} MiB")
flush(None)
W = sum(n for *_, n in sample); rb, sv = (sum(e * n for e, _, n in sample) / W, sum(e * n for _, e, n in sample) / W) if W else (0, 0)
worse = sum(e > s for e, s, _ in sample)
log(f"SAMPLE {len(sample)} tensors: rebuilt {100 * rb:.3f}% vs served {100 * sv:.3f}% error ({100 * (rb / sv - 1):+.1f}%), rebuilt worse on {worse}")
ok = W and rb < sv and worse <= 0.05 * len(sample)
log("REBUILD_NODE", "PASS" if ok else "FAIL", f"in {(time.time() - t0) / 60:.1f} min"); sys.exit(0 if ok else 1)
