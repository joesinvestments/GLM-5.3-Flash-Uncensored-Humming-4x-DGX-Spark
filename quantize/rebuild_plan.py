# Plan the Spark rebuild: which BF16 shards each node downloads and which fused groups it quantizes. Groups never split
# across nodes (fused partners share one global scale); a node also fetches any extra shard its groups reach into.
# Input: layout-base index + BF16 index. Output: plan.json. Usage: [NODES=a,b,c,d] python3 rebuild_plan.py <base index> <bf16 index> <out>
# NODES names the machines that share the work (default: one machine, "local"); each gets a contiguous range of BF16 shards.
import json, os, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rebuild_core import groups

served, bf16, out = (json.load(open(sys.argv[1]))["weight_map"], json.load(open(sys.argv[2]))["weight_map"], sys.argv[3])
Q = sorted(n for n in served if n.endswith(".weight") and n[:-7] + ".weight_scale" in served)
missing = [n for n in Q if n not in bf16]
assert not missing, f"{len(missing)} quantized weights have no BF16 source, e.g. {missing[:3]}"
G = groups(Q)
NODES = os.environ.get("NODES", "local").split(",")
TOTAL = int(next(iter(bf16.values())).split("-of-")[1].split(".")[0])
PER = -(-TOTAL // len(NODES))
node_of_shard = lambda f: NODES[min(len(NODES) - 1, (int(f.split("-")[1]) - 1) // PER)]  # contiguous shard ranges per machine
plan = {n: {"shards": set(), "groups": []} for n in NODES}
for g in G:
    owner = node_of_shard(bf16[g[0]]); plan[owner]["groups"].append(g)
    plan[owner]["shards"].update(bf16[m] for m in g)
split = sum(len({bf16[m] for m in g}) > 1 for g in G)
for n in NODES: plan[n]["shards"] = sorted(plan[n]["shards"])
json.dump({"nodes": plan, "quantized_weights": len(Q), "groups": len(G), "groups_spanning_shards": split}, open(out, "w"), indent=1)
print(f"{len(Q)} quantized weights in {len(G)} groups ({split} span two shards); per node: " +
      ", ".join(f"{n} {len(p['shards'])} shards / {len(p['groups'])} groups" for n, p in plan.items()))
