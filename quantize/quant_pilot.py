# Quantization pilot (Joe, 2026-09-23: build our own checkpoint from Blackfrost's BF16 if it's worth it).
# For real GLM-5.3-Flash DERISKED weights (BF16 source shards), compare NVFP4 reconstruction error of:
#   ours       = the checkpoint we serve (Blackfrost ModelOpt NVFP4 experts + Tony's NVFP4 attention), dequantized
#   minmax     = LLM Compressor 0.14 NVFP4A16 (weight-only, group 16, FP8 scales), min-max observer
#   mse_e1.0 / mse_e1.5 / mse_e2.0 = the same with the MSE observer at expand 1.0, 1.5, 2.0 (1.5 ~ the "4 over 6" trick)
# Metric: relative Frobenius error ||W_hat - W|| / ||W|| per tensor, then the mean per family (kda_attn, mla_attn, experts).
# Only tensors that ours quantizes are compared. Env: BF16_DIR, OURS_DIR, MAX_EXPERT_TENSORS (default 24), DEVICE (cuda; cpu on the Mac), THREADS.
import json, os, re, statistics, sys, time, torch
from collections import defaultdict
from safetensors import safe_open
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from llmcompressor.entrypoints.model_free.lifecycle import calibrate_weight, initialize_quantized_linear

BF16, OURS = os.environ.get("BF16_DIR", "/work/bf16"), os.environ.get("OURS_DIR", "/ours")
MAX_EXP = int(os.environ.get("MAX_EXPERT_TENSORS", "24")); dev = os.environ.get("DEVICE", "cuda")
if dev == "cpu": torch.set_num_threads(int(os.environ.get("THREADS", "8")))  # CPU: leave cores for other work
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
ours_map = json.load(open(f"{OURS}/model.safetensors.index.json"))["weight_map"]

def ours_deq(name, ref):
    """Dequantize our ModelOpt NVFP4 tensor; try both nibble orders and keep the one matching the source."""
    base = name[: -len(".weight")]
    with safe_open(f"{OURS}/{ours_map[name]}", "pt", device=dev) as f:
        q, s, s2 = f.get_tensor(name), f.get_tensor(base + ".weight_scale"), f.get_tensor(base + ".weight_scale_2")
    lo, hi = FP4[(q & 0x0F).long()], FP4[(q >> 4).long()]
    scale = s.float().repeat_interleave(16, dim=1) * s2.float()
    best = None
    for pair in ((lo, hi), (hi, lo)):
        w = torch.stack(pair, dim=-1).reshape(q.shape[0], -1) * scale
        e = rel(w, ref)
        if best is None or e < best: best = e
    return best

def rel(w_hat, w): return ((w_hat.float() - w.float()).norm() / w.float().norm()).item()

def llmc(w, observer, kwargs):
    scheme = preset_name_to_scheme("NVFP4A16", ["Linear"])
    scheme.weights.observer = observer; scheme.weights.observer_kwargs = dict(kwargs)
    m = initialize_quantized_linear(w, scheme, dev); calibrate_weight(m)
    return rel(forward_quantize(m, m.weight, "weight", m.quantization_scheme.weights), w)

METHODS = {"minmax": ("minmax", {}), "mse_e1.0": ("mse", {"expand": 1.0}), "mse_e1.5": ("mse", {"expand": 1.5}), "mse_e2.0": ("mse", {"expand": 2.0})}
MLA_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45}  # from the BF16 index (q_a_proj / kv_a_proj_with_mqa)

results, n_exp = defaultdict(lambda: defaultdict(list)), 0
shards = sorted(f for f in os.listdir(BF16) if f.endswith(".safetensors"))
print("shards:", shards, flush=True)
for shard in shards:
    with safe_open(f"{BF16}/{shard}", "pt", device="cpu") as f:
        for name in f.keys():
            if not name.endswith(".weight") or (name[: -7] + ".weight_scale") not in ours_map: continue
            if ".experts." in name:
                if n_exp >= MAX_EXP: continue
                n_exp += 1
            w = f.get_tensor(name).to(dev)
            if w.ndim != 2 or w.shape[1] % 16: continue
            t0 = time.time(); row = {"ours": ours_deq(name, w)}
            for label, (obs, kw) in METHODS.items(): row[label] = llmc(w, obs, kw)
            layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
            fam = "experts" if ".experts." in name else ("mla_attn" if layer in MLA_LAYERS else "kda_attn")
            for k, v in row.items(): results[fam][k].append(v)
            print(f"{name.split('layers.')[1]:42} {tuple(w.shape)} " + " ".join(f"{k}={100*v:.3f}%" for k, v in row.items()) + f" ({time.time()-t0:.1f}s)", flush=True)
print("\nMEAN relative error per family (lower is better):")
for fam, d in results.items():
    base = statistics.mean(d["ours"])
    print(f"  {fam:12} n={len(d['ours']):3} " + " ".join(f"{k}={100*statistics.mean(v):.3f}% ({100*(statistics.mean(v)/base-1):+.1f}% vs ours)" for k, v in d.items()))
