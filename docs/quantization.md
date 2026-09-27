# Quantizing GLM-5.3-Flash DERISKED yourself

> **Update 2026-09-27, a note from Joe:** I recommend you stick with the Blackfrost weights (Blackfrost's published NVFP4 with Tony's NVFP4 attention, step 2 of the README quick start). This requantization scored better on every short test below, but GLM has been working much better with the factory Blackfrost weights in my long agentic sessions, so production runs those again. The recipe stays here for anyone who wants to experiment.

**A note from Joe.** I do not know if I can distribute weights made from Blackfrost's BF16 master: Blackfrost licenses it commercially and reviews access by hand. So instead of the weights, here is the formula to make them yourself. If you need help, open an issue or reach out to me.

The checkpoint behind the 2026-09-24 to 2026-09-26 numbers in this repo is our own NVFP4 quantization of Blackfrost's [GLM-5.3-Flash-DERISKED-BF16](https://huggingface.co/Blackfrost-AI/GLM-5.3-Flash-DERISKED-BF16). The scripts in [`quantize/`](../quantize) are the exact code that built it.

## What it does

Every 4-bit weight is quantized again from the BF16 master with [LLM Compressor](https://github.com/vllm-project/llm-compressor) 0.14's MSE scale search (NVFP4, weights only, one FP8 scale per 16 weights, `expand` 1.0), then written back in the same ModelOpt layout as the checkpoint it replaces. Everything else in that checkpoint (the layers kept in BF16, norms, config, tokenizer, templates) is copied unchanged, so the result is a drop-in for the build in this repo.

- Fused layers share one global scale, the way vLLM's loader needs: `gate_proj` with `up_proj`, and the KDA input projections `q, k, v, b, f_a, g_a`.
- Why MSE at `expand` 1.0: a pilot on 47 real tensors (0.80B parameters) measured weight reconstruction error against the BF16 master. MSE `expand` 1.0 cut it 6.4% against the conversion we served before, and beat it on every tensor (3.0% to 7.8% each). `expand` 1.5 gave 1.3%, `expand` 2.0 was 9.3% worse, and min-max was 0.2% worse. `quantize/quant_pilot.py` is that pilot.
- Result, same server, same 40 WikiText-2 chunks (72,932 tokens): NLL/token 1.26587 against 1.29080 for the conversion it replaced. GSM8K (250 problems, thinking off) read 99.2% against 98.4% in that test (98.4% in later runs). The same agent tasks finished in 0.80x the time, because fewer answers ran into the length limit. Speed was the same.

## What you need

1. **The BF16 master.** Request licensed access from Blackfrost on the [model page](https://huggingface.co/Blackfrost-AI/GLM-5.3-Flash-DERISKED-BF16) and download it (643 GB, 120 shards plus the index). Its license terms are between you and Blackfrost.
2. **The layout base.** The checkpoint whose layout the new weights go into: Blackfrost's MIT [GLM-5.3-Flash-DERISKED-NVFP4](https://huggingface.co/Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4) with its attention converted to NVFP4 by Tony's recipe, the checkpoint step 2 of the README quick start builds. About 182 GB.
3. **Python with LLM Compressor.** We ran inside `vllm/vllm-openai:v0.30.0` (torch 2.13.0, transformers 5.17.0, safetensors 0.8.0) plus these, installed with `--no-deps` so torch and transformers stay untouched:
   ```bash
   pip install --no-deps llmcompressor==0.14.0 compressed-tensors==0.19.0 loguru==0.7.3 datasets==5.0.1 \
     multiprocess==0.70.19 pandas==3.0.6 pyarrow==25.0.1 xxhash==4.0.1
   ```
4. **Room:** the output is another 182 GB. A GPU is much faster; a CPU works too.

## Steps

```bash
export BF16=/path/to/GLM-5.3-Flash-DERISKED-BF16   # the BF16 master (with model.safetensors.index.json)
export OURS=/path/to/layout-base                    # the layout base from "What you need" step 2
export OUT=/path/to/new-checkpoint                  # empty directory for the result
export PIECES=/path/to/pieces                       # scratch, outside OUT (only used when several machines share the work)

# 1. Plan: which fused groups to quantize (one machine by default)
python3 quantize/rebuild_plan.py $OURS/model.safetensors.index.json $BF16/model.safetensors.index.json plan.json

# 2. Quantize, in bounded chunks (a fresh process every ~4 files keeps memory flat; rerun it to resume)
NODE=local PLAN=plan.json DEVICE=cuda python3 quantize/rebuild_chunks.py      # DEVICE=cpu works, much slower

# 3. Copy what is not re-quantized: config, tokenizer, templates, and the weight files with no 4-bit tensors
rsync -a --ignore-existing $OURS/ $OUT/

# 4. Verify against the layout base before you serve it
python3 quantize/rebuild_verify.py $OURS $OUT
```

Run step 3 only after step 2 passes: the quantizer treats any weight file already in `$OUT` as done.

Each step prints one line to look for: `REBUILD_CHUNKS PASS`, then `REBUILD_VERIFY PASS`. The quantizer checks every new tensor's dtype and shape against the layout base, checks that fused groups share one global scale, and compares a sample of tensors with the BF16 master: the run fails if the new weights are not more accurate than the ones they replace. The verifier checks that every file and tensor is present, that every 4-bit weight changed, and that everything else is byte-identical.

Then serve `$OUT` with the build in this repo (`MODEL=` in `launch/launch_node.sh`, plus the chat template from `template/`).

**Several machines.** Set `NODES=a,b,c,d` for the plan, run step 2 on each machine with `NODE=` set to its name (each only needs its range of BF16 shards), gather every machine's `$PIECES` into one directory, then run `python3 quantize/rebuild_shared.py` there with `PLAN`, `OURS`, `OUT` and `PIECES` set, to join the few files that span machines.

## What to expect

- On four DGX Sparks (GB10 GPUs), our run took about 25 minutes per Spark, all four in parallel.
- On CPU, one shard took 2.6 to 5.2 minutes on 8 threads, so a full CPU run takes many hours.
- A CPU rebuild of two shards passed every check with the same error reduction (7.1% and 6.4% on sampled tensors), but its bytes differ from our GPU build. Expect the same quality, not identical files.
