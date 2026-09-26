# Rebuild core: NVFP4 from Blackfrost's BF16 master. Quantizes BF16 Linear weights with LLM Compressor
# 0.14's NVFP4A16 MSE scale search (expand 1.0; the quant pilot measured -6.4% weight error vs what we serve) and writes them in
# the exact ModelOpt layout we serve, so the result is a drop-in for the loader:
#   weight          uint8 [N, K/2]            two FP4 e2m1 codes per byte, low nibble = even column (served: 9.092% vs 141%)
#   weight_scale    float8_e4m3fn [N, K/16]   same numbers as compressed-tensors' group scale
#   weight_scale_2  float32, served shape     = 1 / compressed-tensors' global scale; shape () for experts, (1,) elsewhere
# Fused groups share one global scale, exactly as the served checkpoint does and as vLLM's fused loaders need: per parent module,
# {gate_proj, up_proj} and the KDA input projections {q, k, v, b, f_a, g_a}_proj; everything else stands alone. The shared scale
# comes from LLM Compressor's own FusionHandler, the same sequence its model_free converter runs.
import re, sys, torch
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from llmcompressor.entrypoints.model_free.lifecycle import calibrate_weight, initialize_quantized_linear
from llmcompressor.modifiers.quantization.calibration import (apply_calibration_status, freeze_module_quantization,
                                                              initialize_observer, observe, update_qparams)
from llmcompressor.observers import FusionHandler

MAG = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])  # e2m1 magnitudes; code = magnitude index, | 8 when negative
FUSED = [("gate_proj", "up_proj"), ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")]

def groups(names):
    """Quantized weight names -> list of fused groups (lists of names), per the served fusion rules."""
    by_parent, out = {}, []
    for n in names:
        parent, leaf = n[: -len(".weight")].rsplit(".", 1)
        by_parent.setdefault(parent, {})[leaf] = n
    for parent, leaves in by_parent.items():
        used = set()
        for fused in FUSED:
            members = [leaves[l] for l in fused if l in leaves]
            if len(members) > 1: out.append(members); used.update(members)
        out += [[n] for n in leaves.values() if n not in used]
    return out

def _pack(m):
    fq = forward_quantize(m, m.weight, "weight", m.quantization_scheme.weights).float()   # LLMC's own dequantized result
    s8, g = m.weight_scale.to(torch.float8_e4m3fn), m.weight_global_scale.float().reshape(1)
    assert torch.equal(s8.float(), m.weight_scale.float()), "group scales are not fp8-exact"
    step = (s8.float() / g).repeat_interleave(16, 1)
    v = torch.where(step > 0, fq / step.clamp_min(1e-30), torch.zeros_like(fq))        # FP4 values, up to bf16 rounding
    mag = (v.abs().unsqueeze(-1) - MAG.to(v.device)).abs().argmin(-1)
    miss = (v.abs() - MAG.to(v.device)[mag]).abs().max().item()                         # < 0.25 decides the grid point
    code = (mag | ((v < 0).long() << 3)).to(torch.uint8)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), s8.contiguous(), (1.0 / g).contiguous(), miss

def quantize_group(ws, dev="cuda"):
    """{name: BF16 [N, K]} for one fused group -> {name: (weight uint8, weight_scale fp8, weight_scale_2 fp32 [1], grid miss)}."""
    scheme = lambda: preset_name_to_scheme("NVFP4A16", ["Linear"])
    mods = {}
    for n, w in ws.items():
        s = scheme(); s.weights.observer = "mse"; s.weights.observer_kwargs = {"expand": 1.0}
        mods[n] = initialize_quantized_linear(w.to(dev), s, dev)
    if len(mods) == 1:
        calibrate_weight(next(iter(mods.values())))
    else:
        for m in mods.values(): initialize_observer(m, "weight"); apply_calibration_status(m)
        FusionHandler.fuse([(m.weight_observer, m) for m in mods.values()])
        observe(mods.values(), base_name="weight"); update_qparams(mods.values(), base_name="weight")
        for m in mods.values(): freeze_module_quantization(m)
    return {n: _pack(m) for n, m in mods.items()}

def dequant(q, s, s2):
    """Served ModelOpt layout -> float32, the way the loader reads it."""
    t = torch.cat([MAG, -MAG]).to(q.device); lo, hi = t[(q & 0x0F).long()], t[(q >> 4).long()]
    return torch.stack((lo, hi), -1).reshape(q.shape[0], -1) * s.float().repeat_interleave(16, 1) * s2.float()
