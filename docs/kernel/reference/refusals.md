# What New refuses

<p class="lede">Every reason <code>New</code> returns an error instead of a kernel, grouped
by why. Each error message names the setting at fault and, where there is one, the setting to
change. The engine's reasons cite the vLLM v0.31.0 source in the code beside them.</p>

## The documents are malformed

- A scenario file has a key its document type does not define. `LoadBundle` decodes strictly,
  so `Open`, `OpenPool` and `OpenInputs` fail here.
- A document fails the contracts' field-level validation: a missing required field, a value
  out of range. Rule-level findings, which depend on the engine version, do not stop
  construction.
- The deployment has no pool at the requested index, or a tensor- or data-parallel width
  below one.
- No engine rules are given. Opening a scenario file (`Open`, `OpenPool`, `OpenInputs`) fails
  earlier, when `Repos.Rules` is unset or has no rules for the engine version it names.
- The engine rules cannot answer: a rules method panics while the kernel is built, as a
  forwarding wrapper around nothing does. The refusal names the panic. A rules method that
  panics later, inside a pricing call, is a defect in the caller's rules and panics there.

## The kernel implementations that run are malformed

- `Inputs.Implementations` lists a primitive with no implementation, or has an implementation with no library, a code hash that is not
  lower-case hexadecimal of at least 12 digits, a code hash with no variant, or two different code
  hashes for one library and variant. A pool runs one build of a kernel.
- A coefficient the cost model needs lists kernel implementations, the pool's implementations for
  the primitive it prices are not all among them (or none is stated for that primitive), and no
  less trusted entry of its name covers the deployment. The error names each refused entry, the
  implementation it is valid for, and the primitive and what runs it.

## The engine would not start it

These are layouts vLLM v0.31.0 refuses at startup.

**Context parallelism**

- Prefill-context parallelism on a model with any layer whose attention is not latent, or
  with a recurrent layer.
- Prefill- and decode-context parallelism together on anything but DSA sparse attention.
- The `a2a` decode-context combine together with prefill-context parallelism.
- A decode-context combine backend the engine release does not accept.
- With decode-context parallelism on, a stated `cp_kv_cache_interleave_size` larger than the
  block size or not dividing it, unless a NIXL connector is configured. With a
  `MultiConnector`, whose child connectors a deployment cannot name, an interleave that does
  not fit, or one left unstated, is refused because the outcome cannot be known.

**Sparse attention (DeepSeek sparse attention, DSA: DeepSeek-V3.2, GLM-5)**

- A stated block size that is not a multiple of 64.
- A chip older than Hopper, where no sparse-attention backend runs.
- A cache data type no backend on the chip serves under the requested parallelism.
- Decode-context parallelism on FlashMLA's sparse backend with a 16-bit cache, with the
  `a2a` combine, with local and gathered head counts that pad to different kernel sizes, or,
  without prefill-context parallelism, with 32 or more attention heads per rank; or prefill-
  and decode-context parallelism there on a cache other than `fp8_ds_mla`.

**Cache data types**

- A cache data type vLLM does not accept, or a recurrent-state data type it does not accept.
- `fp8_inc`, which no CUDA attention backend serves.
- The packed `fp8_ds_mla` and `nvfp4_ds_mla` layouts on a model whose target stack holds any
  attention that is not sparse.
- Plain `nvfp4` on a model with latent attention.

**Block size and architecture**

- A block size a DeepSeek-V4-family model's sparse backends do not accept: DeepSeek-V4.1's
  run 64-token kernel blocks on SM90 and 128-token ones on SM100, and DeepSeek-V4's run
  256-token blocks on every part.
- A model graph that states facts by GPU generation, on a chip whose compute capability the
  catalog does not state or whose generation the graph does not describe.

**Experts**

- An expert-parallel width larger than the model's number of experts, which would leave some
  ranks with none.

**Attention heads**

- A tensor-parallel width that does not divide the KV heads of an attention whose graph says
  its projection cannot replicate a head across ranks (`kv_heads_not_replicated`), as Muse
  Spark's fused QKV and output-gate projection cannot: TP above 16 on its 16 KV heads.

## The kernel cannot price it truthfully

- Pipeline parallelism beside data parallelism or context parallelism (`pcp` or `dcp`
  above one): the stage split prices one pipeline per pool. Pipeline parallelism on a model
  whose graph states its caches, whose sizes are per layer kind over the whole stack rather
  than per stage. A drafter that reads an auxiliary hidden state of a layer outside the
  target's stack.
- `dcp_q_replicate` where it would take effect: decode-context parallelism without
  prefill-context parallelism, on a latent-attention model. It trades a collective for a
  replicated projection the kernel cannot identify in the graph, so pricing only the saving
  would make it look free.
- A cache data type vLLM accepts whose page layout the kernel does not size: the `turboquant`
  and per-token-head formats.
- A quantization, CUDA graph mode or decode-context combine backend the kernel does not
  price.

## A setting selects a kernel the registry has not measured

A setting that changes which kernel runs changes a step's time, so the kernel prices a stated
value from a coefficient measured on that kernel or refuses it with the measurement it needs
(design VLM-3, VAL-8). It never prices a stated value as if it were unstated. Stating the
engine's own default is not a choice: `linear_backend` `auto`, `attention_config.backend`
`auto`, decoder replay on, no custom ops and no replay capture sizes price as unstated.

FlashInfer's trtllm-gen prefill is the one kernel the registry has a measurement of: on b200,
b300, gb200-nvl72 and gb300 it holds a
[pair measured on that kernel](../../registry/reference/sets/cost-model-attention.md) for the
non-latent layers' prefill, `attention_prefill_*_trtllm_gen_kv_bf16` and `_kv_fp8`. The kernel
does not work out from the settings whether an engine runs it: which kernel runs is the engine's
to decide and to report (design X2, SUB-6). The caller reports the implementation of each
lowered primitive in `Inputs.Implementations`, and the lane prices the prefill of the non-latent
layers only when one pair lists exactly the implementation reported for `attention.prefill`
(library, variant and code hash, VAL-8). The pair that lists it prices; the cache dtype, the
block size and `use_trtllm_attention` do not choose it. Latent (MLA) layers keep the part's
pooled pair, and decode is not priced from the lane.

`use_trtllm_attention` is checked against what is reported, not read to select a kernel. Stated
`true` with no pair listing the reported prefill implementation is refused. Stated `false` is
refused when a pair lists the reported implementation (the setting contradicts it) and, for a
model with a non-latent layer, whenever nothing measured the native kernel. A model with no
non-latent layer has nothing for the lane to price, and the setting is accepted. Two pairs
listing the reported implementation are refused, since the cache dtype it serves is then unknown.
Unstated, with no pair listing it, the pooled pair prices the prefill and the assumption
`trtllm_gen_prefill_lane` records why.

The lookup binds each pair to the primitive its name prices. A pair valid for a decode kernel A
does not price a prefill that runs B, a pool that reports A for decode and nothing for prefill
does not admit it, and a prefill that runs two kernels is priced only by a pair that lists both.

The registry's pairs are `transferred` from vLLM 0.25.0 and list no implementation, because no
engine release publishes kernel hashes yet (`kernel_impls.json`; inference-sim/blis-registry#73).
So on the committed registry the lane is not priced: a stated `true` is refused naming that hash,
and an unstated setting keeps the pooled pair. The lane is dormant until an entry lists the
hash of the prefill kernel and blis-vllm reports it.

Every other stated value of the settings below is refused. The middle column is the measurement
that would price it, and the right-hand column the open issue that tracks it. The page and the
code agree, setting by setting and issue by issue (`TestRefusalsPageListsTheUnpricedSettings`).

<!-- unpriced settings -->

| Setting | What would price it | Tracked in |
|---|---|---|
| `attention_config.backend` | per-backend decode and prefill attention timings on the chip, each backend at the cache dtypes it serves | inference-sim/blis-registry#72 |
| `attention_config.mla_prefill_backend` | attention-only MLA prefill timings for FLASH_ATTN, FLASHINFER and TRTLLM_RAGGED, each at a bf16 and an fp8 query and at zero and non-zero cached prefix | inference-sim/blis-registry#67 |
| `kernel_config.linear_backend` | GEMM timings of the kernel classes the backend selects over the registry's token-count grid | inference-sim/blis-registry#66 |
| `compilation_config.custom_ops` | norm, activation and rotary op timings run as custom kernels and as the compiled form | inference-sim/blis-registry#71 |
| `compilation_config.cudagraph_decoder_replay` | decode steps of a model with decoder-replay layers, replay graphs on and off | inference-sim/blis-registry#69 |
| `compilation_config.decoder_replay_cudagraph_capture_sizes` | decode steps at each replay graph size and eager | inference-sim/blis-registry#69 |
| `speculative_config.kv_cache_dtype` | the drafter's attention backend timed at the stated cache dtype and at the target's | inference-sim/blis-registry#68 |
| `attention_config.minimax_m3_msa_decode_backend` | decode timings of the MiniMax-M3 sparse (MSA) attention for each decode backend | inference-sim/blis-registry#72 |
| `attention_config.backend_per_kind` | per-backend attention timings for each KV-cache group kind the override names | inference-sim/blis-registry#72 |
| `kernel_config.linear_backend_per_quant` | GEMM timings of the kernel class each override selects for its scheme | inference-sim/blis-registry#66 |
| `attention_config.use_trtllm_attention` (false) | FlashInfer's native attention timed at the engine's pin, kernel named in each row | inference-sim/blis-registry#72 |

The environment variables `VLLM_FLASHINFER_ALLREDUCE_BACKEND` and `VLLM_MARLIN_USE_ATOMIC_ADD`
(inference-sim/blis-registry#70) reach the kernel through no field, and the engine model refuses them; nothing here prices them.
`prefill_schedule_interval` and `disable_hybrid_kv_cache_manager` are the engine model's, not
the kernel's.

## A stated quantization the checkpoint does not declare

vLLM v0.32.0 requires a stated `--quantization` to equal the method the checkpoint declares,
after the method classes' overrides (`vllm/config/model.py:1370-1449`). A stated value that
names the checkpoint's own method therefore changes nothing, and the kernel prices it as an
unstated one. Shown from the graph's stored formats, since the graph does not carry the
declared method:

- `mxfp8`, when the graph stores its weights at FP8 with the microscaling layout, one 1-byte
  scale per 32 elements (the nodes the checkpoint leaves unquantized carry their own dtype),
  The graph states no block shape or declared method; carrying them is the catalog deriver's,
  and until it does a 32 x 32 FP8 checkpoint with the same resident scale would pass. The one
  such family in the catalog, DeepSeek-V4, is overridden to `deepseek_v4_fp8` by vLLM whatever
  is stated, and the kernel records that override. A bf16 checkpoint, or FP8 with other scales, is refused: vLLM would quantize it online or
  refuse the mismatch, and neither is priced. `modelopt_mxfp8` is refused, because vLLM reaches
  it only from a ModelOpt block and the graph cannot say which kind of MXFP8 checkpoint this
  is; state `mxfp8`.
- `gpt_oss_mxfp4` and `mxfp4`, when the graph is gpt-oss with MXFP4 routed experts and
  everything else unquantized. vLLM resolves `mxfp4` to `gpt_oss_mxfp4` for that checkpoint,
  and the kernel records that as an override. Either on any other graph is refused: vLLM
  refuses the mismatch with the declared method or quantizes online.
- `modelopt_mixed` is refused. It names a ModelOpt MIXED_PRECISION checkpoint, whose per-layer
  algorithms the catalog deriver does not place, so no graph shows the layout.

## A coefficient is missing

- A required coefficient has no entry in the sets the scenario names, or none whose scope
  covers this chip, model and layout: the matrix-multiply efficiency for the served data
  type, the memory derate, the memory magnitudes, or a collective's floor, rate or transition
  rate at the width the layout needs.
- A collective's group width is not one the registry measured on the chip. The kernel does
  not borrow a narrower width's figures, because they would understate the wider group. Two
  cases are priced rather than refused: a group narrower than every measured width, at the
  narrowest, and a group wider than 16, the widest the registry's sweeps cover, at the
  chip's widest measured width.

Optional coefficients are not refused when missing; see
[Provenance, Evidence, Resolved](provenance.md#optional-coefficients-that-were-missing).

## In the source

The checks run in `New` and the functions it calls in
[`new.go`](https://github.com/inference-sim/blis-substrate/blob/main/kernel/new.go),
[`sparse_mla.go`](https://github.com/inference-sim/blis-substrate/blob/main/kernel/sparse_mla.go)
and [`internal/resolve`](https://github.com/inference-sim/blis-substrate/tree/main/kernel/internal/resolve),
and the tests in `refusals_test.go`, `sparse_mla_test.go` and `dcp_knobs_test.go` check most of
them.
