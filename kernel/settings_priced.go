package latencykernel

import (
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// The kernel-facing settings (VLM-3, VAL-8, design §11.2): settings that change which kernel
// runs, and so a step's time. A stated value is priced from a coefficient measured on that
// kernel, or refused with the measurement it needs. It is never priced as if it were
// unstated: the kernel prices the engine's own choice only when the setting is unstated.
//
// Two things sit here. unpricedSettings lists the settings whose every stated value is
// refused, and what would price them. applyAttentionLane prices the one kernel the registry has
// a measurement for, FlashInfer's trtllm-gen prefill, and only where the caller reports that
// kernel as the implementation of the prefill primitive and the registry shows its pair valid for
// exactly that implementation (VAL-8). Which kernel an engine runs for a setting is the engine's
// to decide and to report; this package does not infer it from the settings.

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
// non-latent layer when the lane is priced (applyAttentionLane), the part's pooled pair
// otherwise. A latent layer is a different kernel and keeps the pooled pair.
func (k *Kernel) prefillFor(kind model.AttentionKind) prefillPair {
	if k.laneRuns && !latentAttention(kind) {
		return k.lanePair
	}
	return prefillPair{k.attentionPrefillFloor, k.attentionPrefillScale}
}

// The trtllm-gen lane is chosen by the kernel implementation the caller reports for the prefill
// primitive, and by nothing the settings say. The engine (blis-vllm's lowering) decides which
// kernel runs; this package prices the implementation it is told runs (design X2, SUB-6). A
// pair is priced only when the registry shows it valid for exactly that implementation (VAL-8):
// the lookup returns an entry keyed to it as exact, and every other entry as unkeyed or not at
// all. Two things are read from the settings, and neither selects a kernel: use_trtllm_attention
// is checked against the reported implementation, and a stated false that cannot be priced is
// refused as every other unpriced setting is.

// lanePairs are the cache dtypes of the lane's entries. Which one prices is the one whose
// entries are keyed to the reported prefill implementation; the cost model does not derive it
// from the engine's cache dtype.
var lanePairs = []string{"bf16", "fp8"}

// applyAttentionLane prices the prefill of the non-latent attention layers on the trtllm-gen
// lane when, and only when, the caller reported the implementation of the attention.prefill
// primitive (Inputs.Implementations) and one pair of the lane lists exactly that implementation
// (the lookup returns it as exact). The registry's lane is transferred from vLLM 0.25.0 and lists
// none, so on it the lane is not priced.
//
//   - No pair is exact and use_trtllm_attention is stated true: refused, saying why (the pairs
//     list no implementation, list another, or the caller reported none).
//   - No pair is exact and the setting is unstated: the part's pooled pair prices the prefill,
//     and the reason is recorded as an assumption.
//   - A pair is exact and the setting is stated false: refused, because the setting contradicts
//     the implementation reported.
//   - Both pairs are exact: refused, because the entries do not say which cache dtype the
//     implementation serves.
//
// A stated false with a non-latent layer and no exact pair is refused: nothing measured the
// native kernel. A model with no non-latent layer has nothing for the lane to price, and the
// setting is accepted.
//
// The match is by the primitive the lane's names bind to (attention.prefill), so a decode-only
// trtllm-gen implementation cannot admit the pair. When the plan reports several prefill
// implementations, a pair is exact only if it lists every one of them: one value cannot price a
// prefill that runs two kernels.
func (k *Kernel) applyAttentionLane(c *resolve.Coefficients, used map[model.AttentionKind]bool) error {
	plain := false
	for kind, u := range used {
		plain = plain || (u && !latentAttention(kind))
	}
	if !plain {
		return nil
	}
	stated := k.engine.UseTRTLLMAttention
	var exact []string
	var why []string
	for _, dtype := range lanePairs {
		floor, scale := trtllmGenPrefillFloor+dtype, trtllmGenPrefillScale+dtype
		if r := laneRefusedByImplementation(c, floor, scale); r != "" {
			why = append(why, r)
			continue
		}
		if !c.Has(floor) || !c.Has(scale) {
			continue
		}
		pairExact := true
		for _, name := range []string{floor, scale} {
			if r, _ := c.Admitted(name); r.Match != coefficient.KernelExact {
				pairExact = false
				why = append(why, fmt.Sprintf("%s states no kernel implementation it is valid for (kernel_impls), so it "+
					"applies to every build of the kernel and nothing shows it was measured on, or moved to, the one the "+
					"engine runs", name))
				break
			}
		}
		if pairExact {
			exact = append(exact, dtype)
		}
	}
	switch {
	case len(exact) > 1:
		return fmt.Errorf("the trtllm-gen prefill pairs of cache dtypes %v both list the prefill implementation the "+
			"engine reports (%s), so the kernel cannot tell which cache dtype it serves; each pair must list the "+
			"build it was moved to", exact, describeLaneRunning(c))
	case len(exact) == 1:
		if stated != nil && !*stated {
			return fmt.Errorf("attention_config.use_trtllm_attention is stated as \"false\", and the engine reports "+
				"a prefill implementation (%s) that the trtllm-gen pair of cache dtype %s lists, so the setting "+
				"contradicts the implementation and the kernel will not price either", describeLaneRunning(c), exact[0])
		}
		floor, scale := trtllmGenPrefillFloor+exact[0], trtllmGenPrefillScale+exact[0]
		k.laneRuns = true
		k.lanePair = prefillPair{time.Duration(c.ValueOr(floor, 0) * float64(time.Microsecond)), c.ValueOr(scale, 0)}
		return nil
	}
	// No pair is shown valid for the reported implementation.
	if stated != nil && !*stated {
		return fmt.Errorf("attention_config.use_trtllm_attention is stated as \"false\", and the registry has " +
			"no measurement that prices it, so the kernel will not price it as the default (tracked in " +
			useTRTLLMFalseIssue + "). Needed: " +
			"decode and prefill timings of FlashInfer's native (non-trtllm-gen) attention on the chip, " +
			"collected at the engine's FlashInfer pin with the kernel named in each row; the registry's " +
			"older native-path label does not say which kernel ran")
	}
	if len(why) == 0 {
		why = append(why, "the scenario's coefficient sets carry no trtllm-gen prefill pair for "+k.chip.Name)
		if stated == nil {
			return nil
		}
	}
	reason := strings.Join(why, "; ") + ". Needed: the code hash of the trtllm-gen prefill kernel of the engine's " +
		"FlashInfer pin, which an engine release's kernel_impls.json would publish (inference-sim/blis-registry#73), " +
		"stated on the entries and reported by the engine as the implementation of attention.prefill"
	if stated != nil {
		return fmt.Errorf("attention_config.use_trtllm_attention is stated as \"true\" on %s, and the kernel cannot show "+
			"that the trtllm-gen prefill coefficients are valid for the prefill implementation the engine reports "+
			"(%s; VAL-8), so it will not price it as the default: %s", k.chip.Name, describeLaneRunning(c), reason)
	}
	k.assume("trtllm_gen_prefill_lane", "pooled", "the trtllm-gen prefill pair cannot be shown valid for the prefill "+
		"implementation reported ("+describeLaneRunning(c)+"): "+reason+". The part's pooled pair prices the prefill.")
	return nil
}

// describeLaneRunning is the prefill implementations the caller reported, for a message.
func describeLaneRunning(c *resolve.Coefficients) string {
	if r := c.Implementations()[laneRunningPrimitive]; len(r) > 0 {
		return laneRunningPrimitive + " runs " + coefficient.DescribeImpls(r)
	}
	return "no implementation reported for " + laneRunningPrimitive
}

// laneRunningPrimitive is the lowered primitive the lane's coefficient names bind to.
const laneRunningPrimitive = "attention.prefill"

// laneRefusedByImplementation describes the lane entries the lookup refused because they list
// kernel implementations and the primitive's running set is not covered by them, or "" when it
// refused none.
func laneRefusedByImplementation(c *resolve.Coefficients, names ...string) string {
	var out []string
	for _, r := range c.Refused() {
		if slices.Contains(names, r.Entry.Name) {
			running := "no implementation of " + coefficient.PrimitiveOf(r.Entry.Name)
			if len(r.Running) > 0 {
				running = coefficient.PrimitiveOf(r.Entry.Name) + " runs " + coefficient.DescribeImpls(r.Running)
			}
			out = append(out, fmt.Sprintf("%s is valid for %s, and the engine reports %s", r.Entry.Name,
				coefficient.DescribeImpls(r.Entry.KernelImpls), running))
		}
	}
	if len(out) == 0 {
		return ""
	}
	return strings.Join(out, "; ") + "; a different build or kernel, or a prefill that also runs another, is not priced from it"
}
