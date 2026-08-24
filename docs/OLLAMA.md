# The most efficient Ollama build for AG

The single binding constraint is **VRAM**. A model that fits entirely in the GPU runs
at full speed; one that spills even a little into system RAM drops to a fraction of that
throughput because the CPU has to process the offloaded layers. Everything below follows
from "keep the whole model + its KV cache inside VRAM."

## Sizing rule of thumb

| VRAM | Fully-in-GPU sweet spot (Q4_K_M) | Notes |
|------|----------------------------------|-------|
| 8 GB | **7–8B** (~4.5–5 GB weights) | leaves ~3 GB for KV cache/context — the fast lane |
| 12 GB | 8–13B | 14B fits with a small context |
| 16 GB | 13–14B comfortably | |
| 24 GB+ | 32B (Q4) | serious local quality |

**This machine — RTX 4060, 8 GB VRAM, 16 cores, ~34 GB RAM:**
- **Fast lane (recommended default): `qwen2.5:7b`** — 4.7 GB, fits fully in VRAM, all
  layers on GPU. Already your default. Good general model. Peers: `llama3.1:8b`,
  `mistral:7b`; for code, `qwen2.5-coder:7b`.
- **Quality lane (slower): `qwen3:14b`** — 9.3 GB > 8 GB VRAM, so ~1–2 GB and some layers
  offload to CPU/RAM. Usable and noticeably smarter, but several× slower per token. Reach
  for it on non-interactive, quality-critical runs, not everyday prompts.
- **Avoid on 8 GB:** fp16/Q8 of a 7B (no room left for context), and 32B+ (heavy offload → very slow).

## Quantization

`Q4_K_M` is the balance point (near-Q5 quality, smallest practical size) and is what the
default `:7b`/`:14b` tags already use. Step to `Q5_K_M` only if it *still* fits fully in
VRAM. `Q6`/`Q8`/fp16 are worth it only for small models with VRAM to spare — otherwise the
extra bytes push you into offload and cost more speed than the quality is worth.

```bash
ollama pull qwen2.5:7b            # Q4_K_M by default
ollama pull qwen2.5:7b-instruct-q5_K_M   # a notch up, if it still fits in VRAM
```

## Keep it in VRAM: context + KV cache

The KV cache grows with context length and lives in VRAM. On 8 GB, keep `num_ctx` modest
(4096–8192) so weights **and** cache stay resident. AG's optimizer/critic/reviser messages
are short, so 8192 is plenty of headroom.

Two server-level env vars (set once on the machine running `ollama serve`) buy back VRAM:

```bash
OLLAMA_FLASH_ATTENTION=1     # smaller, faster attention → more room for context
OLLAMA_KV_CACHE_TYPE=q8_0    # quantize the KV cache (newer Ollama) → fit more context
```

## Keep it loaded: `keep_alive`

AG makes **3–4 model calls per run** (optimize → execute → critique → maybe revise). If the
model unloads between them you pay a multi-second reload each time. AG now sends
`keep_alive` (default **30m**) so the model stays resident across a run and between runs.
Set `"-1"` to keep it loaded until you stop Ollama.

## Recommended `config.json` for this rig

```json
{
  "backend": "ollama",
  "ollama_model": "qwen2.5:7b",
  "ollama_keep_alive": "30m",
  "ollama_options": { "num_ctx": 8192, "num_gpu": 99 }
}
```

- `num_gpu: 99` tells Ollama to put *all* layers on the GPU (a no-op safety for the 7B,
  which already fully offloads; drop it or lower it if you switch to a 14B and want to
  control how much stays on the GPU).
- `ollama_options` is a passthrough — any Ollama option (`temperature`, `top_p`,
  `num_thread`, …) can go here and AG forwards it. `num_predict` is always set from
  `max_output_tokens`.

## Verify you're actually GPU-resident

```bash
ollama ps                 # PROCESSOR should read 100% GPU for the fast lane
ollama run qwen2.5:7b --verbose "hi"   # prints eval rate (tokens/s)
python -m ag host         # AG's view: GPU model + VRAM, cores, RAM
```

If `ollama ps` shows a CPU split for a 7B model, something else is using VRAM (close it),
or `num_ctx` is too high — lower it until the split returns to 100% GPU.

## Keeping the model current

Model builds get republished under the same tag. AG checks **on demand only** (never a
background poll): the web app pings once after a run and offers a one-click **Update now?**,
or run it yourself:

```bash
python -m ag update            # check the configured model against the registry
python -m ag update --apply    # pull the newer build now
```
