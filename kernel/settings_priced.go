package latencykernel

import (
	"fmt"
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// The kernel-facing settings (VLM-3, VAL-8, design §11.2): settings that change which kernel
// runs, and so a step's time. A stated value is priced from a coefficient measured on that
// kernel, or refused with the measurement it needs. It is never priced as if it were
// unstated: the kernel prices the engine's own choice only when the setting is unstated.
//
// Two things sit here. unpricedSettings lists the settings whose every stated value is
// refused, and what would price them. applyAttentionLane prices the one stated value the
// registry has a measurement for: attention_config.use_trtllm_attention true, whose prefill
// runs FlashInfer's trtllm-gen kernel (vllm/utils/flashinfer.py:697-789 at 5df63a72).

// The prefill coefficients measured on the trtllm-gen kernel alone, one pair per data-center
// Blackwell part and per KV cache dtype of the sweep (registry/scripts/
// emit_attention_trtllm_gen.py). A part or a dtype without the pair has no measurement.
const (
	trtllmGenPrefillFloor = "attention_prefill_floor_trtllm_gen_kv_"
	trtllmGenPrefillScale = "attention_prefill_work_scale_trtllm_gen_kv_"
)

// unpricedSetting is a kernel-facing setting for which the registry holds no measurement of
// any stated value.
type unpricedSetting struct {
	// Field is the setting as the engine names it.
	Field string
	// Stated returns the stated value and true when the settings state one the kernel would
	// otherwise drop. The engine's own default is not a stated value: stating it asks for
	// the kernel the engine would choose anyway.
	Stated func(enginesettings.Settings) (string, bool)
	// Needs names the measurement that would price a value.
	Needs string
	// Issue is the open issue that tracks the measurement, as owner/repo#number (X28).
	Issue string
}

// unpricedSettings is every kernel-facing setting the kernel refuses whatever its value, in
// the order a refusal reports them. The list may only shrink: a setting leaves it in the
// change that adds the coefficients pricing its values.
var unpricedSettings = []unpricedSetting{
	{
		Field: "attention_config.backend",
		Issue: "inference-sim/blis-registry#72",
		Stated: func(s enginesettings.Settings) (string, bool) {
			return s.AttentionBackend, s.AttentionBackend != "" && !strings.EqualFold(s.AttentionBackend, "auto")
		},
		Needs: "per-backend decode and prefill attention timings on the chip, each backend at the " +
			"cache dtypes it serves; the registry's attention entries are fitted on the " +
			"backend vLLM selects, not on a named one",
	},
	{
		Field: "attention_config.mla_prefill_backend",
		Issue: "inference-sim/blis-registry#67",
		Stated: func(s enginesettings.Settings) (string, bool) {
			return s.MLAPrefillBackend, s.MLAPrefillBackend != ""
		},
		Needs: "attention-only MLA prefill timings, not whole-module ones, for FLASH_ATTN, FLASHINFER " +
			"and TRTLLM_RAGGED, each at a bf16 and at an fp8 query and at zero and non-zero cached " +
			"prefix, on the chip and the model's head layout; the module timings in the registry's " +
			"sources run TRTLLM_RAGGED only at an fp8 query and FLASH_ATTN only at bf16, and time " +
			"FLASHINFER prefill nowhere",
	},
	{
		Field: "kernel_config.linear_backend",
		Issue: "inference-sim/blis-registry#66",
		Stated: func(s enginesettings.Settings) (string, bool) {
			v := strings.ReplaceAll(strings.ToLower(s.LinearBackend), "-", "_")
			return v, v != "" && v != "auto"
		},
		Needs: "GEMM timings of the kernel classes the backend selects over the registry's token-count " +
			"grid on the chip: for flashinfer_cutlass the FlashInfer CUTLASS NvFp4 and FP8 kernels, " +
			"for humming the Humming kernels; the registry's ramps are fitted on the CUTLASS FP8 and " +
			"FlashInfer CuTe-DSL NvFp4 kernel classes only",
	},
	{
		Field: "attention_config.minimax_m3_msa_decode_backend",
		Issue: "inference-sim/blis-registry#72",
		Stated: func(s enginesettings.Settings) (string, bool) {
			// The engine's default is triton (vllm/config/attention.py:43 at 5df63a72).
			return s.MiniMaxM3MSADecodeBackend, s.MiniMaxM3MSADecodeBackend != "" && s.MiniMaxM3MSADecodeBackend != "triton"
		},
		Needs: "decode timings of the MiniMax-M3 sparse (MSA) attention for each decode backend on the chip; " +
			"the registry holds none for the sparse decode kernel",
	},
	{
		Field: "attention_config.backend_per_kind",
		Issue: "inference-sim/blis-registry#72",
		Stated: func(s enginesettings.Settings) (string, bool) {
			return fmt.Sprint(s.AttentionBackendPerKind), len(s.AttentionBackendPerKind) > 0
		},
		Needs: "per-backend decode and prefill attention timings on the chip for each KV-cache group kind " +
			"the override names, as for attention_config.backend",
	},
	{
		Field: "kernel_config.linear_backend_per_quant",
		Issue: "inference-sim/blis-registry#66",
		Stated: func(s enginesettings.Settings) (string, bool) {
			return fmt.Sprint(s.LinearBackendPerQuant), len(s.LinearBackendPerQuant) > 0
		},
		Needs: "GEMM timings of the kernel class each override selects for its quantization scheme, over the " +
			"registry's token-count grid on the chip, as for linear_backend",
	},
	{
		Field: "compilation_config.custom_ops",
		Issue: "inference-sim/blis-registry#71",
		Stated: func(s enginesettings.Settings) (string, bool) {
			return strings.Join(s.CustomOps, ","), len(s.CustomOps) > 0
		},
		Needs: "norm, activation and rotary op timings run as custom kernels and as the compiled " +
			"form, on the chip; the registry times no op both ways",
	},
	{
		Field: "compilation_config.cudagraph_decoder_replay",
		Issue: "inference-sim/blis-registry#69",
		Stated: func(s enginesettings.Settings) (string, bool) {
			// The engine's default is on (vllm/config/compilation.py:559 at 5df63a72), so
			// only a stated off differs.
			return "false", s.CUDAGraphDecoderReplay != nil && !*s.CUDAGraphDecoderReplay
		},
		Needs: "decode steps of a model with decoder-replay layers with the replay graphs on and off, " +
			"on the chip",
	},
	{
		Field: "compilation_config.decoder_replay_cudagraph_capture_sizes",
		Issue: "inference-sim/blis-registry#69",
		Stated: func(s enginesettings.Settings) (string, bool) {
			return fmt.Sprint(s.DecoderReplayCUDAGraphCaptureSizes), len(s.DecoderReplayCUDAGraphCaptureSizes) > 0
		},
		Needs: "decode steps of a model with decoder-replay layers at each replay graph size and " +
			"eager, on the chip; the registry holds no replay-graph launch cost",
	},
	{
		Field: "speculative_config.kv_cache_dtype",
		Issue: "inference-sim/blis-registry#68",
		Stated: func(s enginesettings.Settings) (string, bool) {
			if s.Speculative == nil {
				return "", false
			}
			return s.Speculative.KVCacheDType, s.Speculative.KVCacheDType != ""
		},
		Needs: "the drafter's attention backend timed at the stated cache dtype and at the target's, " +
			"decode and prefill, on the chip; the registry's attention rate counts an fp8 cache as " +
			"half the bytes, and the paired fp8 and bf16 sweeps it is fitted on run at 0.62 of the " +
			"bf16 time at long contexts, not 0.5",
	},
}

// refuseUnpricedSettings refuses the first stated setting of unpricedSettings.
func refuseUnpricedSettings(e enginesettings.Settings) error {
	for _, u := range unpricedSettings {
		if v, stated := u.Stated(e); stated {
			return fmt.Errorf("%s is stated as %q, and the registry has no measurement that prices it, so "+
				"the kernel will not price it as the default (tracked in %s). Needed: %s", u.Field, v, u.Issue, u.Needs)
		}
	}
	return nil
}

// useTRTLLMFalseIssue tracks the native-attention timing a stated false needs.
const useTRTLLMFalseIssue = "inference-sim/blis-registry#72"

// prefillPair is a prefill attention floor and the fraction of the dense ramp it reaches.
type prefillPair struct {
	floor time.Duration
	scale float64
}

// prefillFor is the pair that prices one attention layer's prefill: the trtllm-gen lane's for a
// non-latent layer when the engine runs that kernel (applyAttentionLane), the part's pooled
// pair otherwise. The MLA and sparse-MLA backends never read use_trtllm_attention.
func (k *Kernel) prefillFor(kind model.AttentionKind) prefillPair {
	if k.laneRuns && !latentAttention(kind) {
		return k.lanePair
	}
	return prefillPair{k.attentionPrefillFloor, k.attentionPrefillScale}
}

// trtllmGenRuns reports whether v0.32.0's FlashInfer backend runs the trtllm-gen kernel for the
// prefill of a non-latent layer on a part that has it, from what the engine runs and not from the
// flag alone (vllm/utils/flashinfer.py:697-789 and vllm/v1/attention/backends/flashinfer.py:1429-1433
// at 5df63a72):
//
//   - decode-context parallelism: TRTLLM prefill cannot combine across ranks, so the setting is
//     ignored (utils/flashinfer.py:733-741) and the pooled pair stands;
//   - stated true, or a stated block size of 128 or more (the backend forces it): the kernel runs;
//   - stated false: the native kernel runs, which nothing measured (refused);
//   - unstated, with an auto cache dtype: the kernel runs (utils/flashinfer.py:766). With any
//     other cache dtype and no stated block size the native kernel runs, and is priced at the
//     pooled pair as before. An unstated block size is the engine's and is not read here.
func (k *Kernel) trtllmGenRuns() (runs, stated bool) {
	if k.layout.DCP > 1 {
		return false, false
	}
	if t := k.engine.UseTRTLLMAttention; t != nil {
		return *t, true
	}
	if k.engine.BlockSize >= 128 {
		return true, false
	}
	return k.cacheDType == "" || k.cacheDType == "auto", false
}

// kvLane names the lane's cache dtype for the engine's, or "" where the sweep has none.
func (k *Kernel) kvLane() string {
	switch k.cacheDType {
	case "", "auto", "bfloat16", "float16":
		return "bf16"
	case "fp8", "fp8_e4m3", "fp8_e5m2":
		return "fp8"
	}
	return ""
}

// applyAttentionLane chooses the prefill pair of the non-latent attention layers from what the
// engine runs. A stated true on a part or cache dtype with no measurement of the kernel is
// refused, and so is a stated false where the model has a non-latent layer, whose native kernel
// nothing measured; where the model has none, or decode-context parallelism ignores the setting
// for prefill, a stated true is accepted and changes nothing.
func (k *Kernel) applyAttentionLane(c *resolve.Coefficients, used map[model.AttentionKind]bool) error {
	plain := false
	for kind, u := range used {
		plain = plain || (u && !latentAttention(kind))
	}
	if !plain {
		return nil
	}
	runs, stated := k.trtllmGenRuns()
	if stated && !runs {
		return fmt.Errorf("attention_config.use_trtllm_attention is stated as \"false\", and the registry has " +
			"no measurement that prices it, so the kernel will not price it as the default (tracked in " +
			useTRTLLMFalseIssue + "). Needed: " +
			"decode and prefill timings of FlashInfer's native (non-trtllm-gen) attention on the chip, " +
			"collected at the engine's FlashInfer pin with the kernel named in each row; the registry's " +
			"older native-path label does not say which kernel ran")
	}
	if !runs {
		return nil
	}
	dtype := k.kvLane()
	floor, scale := trtllmGenPrefillFloor+dtype, trtllmGenPrefillScale+dtype
	if dtype == "" || !c.Has(floor) || !c.Has(scale) {
		if stated {
			return fmt.Errorf("attention_config.use_trtllm_attention is stated as \"true\" on %s with cache dtype %q, "+
				"and the scenario's coefficient sets carry no prefill measurement of the trtllm-gen kernel "+
				"there (%s and %s), so the kernel will not price it as the default. Needed: prefill "+
				"attention timings of FlashInfer's trtllm-gen kernel on this chip at this cache dtype; the "+
				"registry has them for b200, b300, gb200-nvl72 and gb300 at bf16 and fp8",
				k.chip.Name, k.cacheDType, floor, scale)
		}
		if k.engine.BlockSize >= 128 && c.Has(trtllmGenPrefillFloor+"bf16") {
			k.assume("trtllm_gen_prefill_lane", "absent", "the engine forces the trtllm-gen prefill at a "+
				"block size of 128 or more, and the registry has no such measurement at cache dtype "+
				k.cacheDType+", so the part's pooled pair prices it")
		}
		return nil
	}
	k.laneRuns = true
	k.lanePair = prefillPair{time.Duration(c.ValueOr(floor, 0) * float64(time.Microsecond)), c.ValueOr(scale, 0)}
	return nil
}
