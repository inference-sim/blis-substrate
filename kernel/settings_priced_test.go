package latencykernel

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// VLM-3, VAL-8: a kernel-facing setting is priced from a measurement of the kernel it selects,
// or refused with the measurement it needs. These tests price MiniMax-M3 (full and
// sliding-window attention, the model whose scored rows state use_trtllm_attention) on the
// data-center Blackwell parts that carry the trtllm-gen prefill lane, and on one that does not.

// laneCase is a built perturbation scenario for the model on the chip, with the engine settings
// the perturbation chose and the tensor-parallel width it priced.
func laneCase(t *testing.T, d *load.Data, chip string) Inputs {
	t.Helper()
	for _, c := range perturbationScenarios(t, d, "minimax-m3") {
		if strings.HasPrefix(c.label, "minimax-m3/"+chip+"/tp4/dp1/ept") || strings.HasPrefix(c.label, "minimax-m3/"+chip+"/tp4/dp1/epf") {
			return c.in
		}
	}
	t.Fatalf("no minimax-m3 scenario builds on %s at tp 4", chip)
	return Inputs{}
}

// quantCase is a built perturbation scenario for the catalog model on the chip.
func quantCase(t *testing.T, d *load.Data, name, chip string) Inputs {
	t.Helper()
	for _, c := range perturbationScenarios(t, d, name) {
		if strings.HasPrefix(c.label, name+"/"+chip+"/tp") {
			return c.in
		}
	}
	t.Fatalf("no %s scenario builds on %s", name, chip)
	return Inputs{}
}

func flag(b bool) *bool { return &b }

// The registry states no kernel hash for the trtllm-gen prefill (none is published), so its lane
// is not priced. These tests key the lane to fixture implementations, in test data only: the
// hashes below are not any real kernel's. The bf16 pair is valid for laneBF16 and the fp8 pair
// for laneFP8, two builds of one library.
var (
	laneBF16 = plan.KernelImpl{Library: "flashinfer", Variant: "fixture-trtllm-prefill-bf16", CodeHash: "0123456789ab"}
	laneFP8  = plan.KernelImpl{Library: "flashinfer", Variant: "fixture-trtllm-prefill-fp8", CodeHash: "123456789abc"}
	// laneDecode is a decode kernel of the same library and the same hash digits as laneBF16: a pool
	// that runs it for decode must not admit a pair that prices prefill.
	laneDecode = plan.KernelImpl{Library: "flashinfer", Variant: "fixture-trtllm-decode", CodeHash: laneBF16.CodeHash}
	// laneSource stands for the build the 0.25.0 sweep ran, which the entries record as the
	// implementation their data came from.
	laneSource = coefficient.KernelImpl{Library: "flashinfer", Variant: "fixture-trtllm-prefill-bf16", CodeHash: "ba9876543210"}
)

func asKey(i plan.KernelImpl) coefficient.KernelImpl {
	return coefficient.KernelImpl{Library: i.Library, Variant: i.Variant, CodeHash: i.CodeHash}
}

// keyedLane states, on the lane's entries of the scenario's sets, that the bf16 pair is valid for
// laneBF16 and the fp8 pair for laneFP8, measured on the fixture source, and reports prefill as
// the implementations of the attention.prefill primitive.
func keyedLane(in Inputs, prefill ...plan.KernelImpl) Inputs {
	return withPrefill(keyLaneEntries(in, map[string]plan.KernelImpl{"bf16": laneBF16, "fp8": laneFP8}), prefill...)
}

// keyLaneEntries keys the pair of each cache dtype to the implementation given for it.
func keyLaneEntries(in Inputs, byDType map[string]plan.KernelImpl) Inputs {
	out := make([]*coefficient.Set, len(in.Coefficients))
	for i, s := range in.Coefficients {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j := range c.Coefficients {
			e := &c.Coefficients[j]
			for dtype, impl := range byDType {
				if slices.Contains(laneNames(dtype), e.Name) {
					e.KernelImpls = []coefficient.KernelImpl{asKey(impl)}
					e.MeasuredOn = []coefficient.KernelImpl{laneSource}
				}
			}
		}
		out[i] = &c
	}
	in.Coefficients = out
	return in
}

// withPrefill reports the implementations of the attention.prefill primitive, leaving the other
// primitives as they were; none removes the primitive.
func withPrefill(in Inputs, impls ...plan.KernelImpl) Inputs {
	return withRunning(in, plan.PrimAttentionPrefill, impls...)
}

// withRunning sets the implementations the pool reports for one primitive (none removes it).
func withRunning(in Inputs, prim plan.Primitive, impls ...plan.KernelImpl) Inputs {
	m := map[plan.Primitive][]plan.KernelImpl{}
	for k, v := range in.Implementations {
		m[k] = v
	}
	if len(impls) == 0 {
		delete(m, prim)
	} else {
		m[prim] = impls
	}
	in.Implementations = m
	return in
}

// laneAssumed reports whether the kernel recorded that the lane was not priced, and why.
func laneAssumed(k *Kernel) (string, bool) {
	for _, a := range k.assumptions {
		if a.Name == "trtllm_gen_prefill_lane" {
			return a.Scope, true
		}
	}
	return "", false
}

// prefillAndDecode prices a prefill step and a decode step separately, each as one comparable
// string.
func prefillAndDecode(k *Kernel) (prefill, decode string) {
	price := func(r kernel.ReqShape) string {
		e := k.StepTime(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{r}})
		return fmt.Sprintf("%d %d", e.Overlap, e.NoOverlap)
	}
	return price(kernel.ReqShape{Scheduled: 2048, PromptLen: 2048}),
		price(kernel.ReqShape{Scheduled: 1, Computed: 4095, PromptLen: 4096})
}

func withEngine(in Inputs, f func(*enginesettings.Settings)) Inputs {
	e := in.Engine.Clone()
	f(&e)
	in.Engine = &e
	return in
}

func mustKernel(t *testing.T, in Inputs) *Kernel {
	t.Helper()
	k, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return k
}

// scaleEntry returns the scenario's sets with every entry of the given name scaled by f.
func scaleEntry(in Inputs, name string, f float64) Inputs {
	out := make([]*coefficient.Set, len(in.Coefficients))
	for i, s := range in.Coefficients {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j := range c.Coefficients {
			if c.Coefficients[j].Name == name {
				c.Coefficients[j].Value *= f
				if iv := c.Coefficients[j].CI95; iv != nil {
					c.Coefficients[j].CI95 = &coefficient.Interval{Low: iv.Low * f, High: iv.High * f}
				}
			}
		}
		out[i] = &c
	}
	in.Coefficients = out
	return in
}

// laneNames are the lane's coefficient names at a cache dtype.
func laneNames(dtype string) []string {
	return []string{trtllmGenPrefillFloor + dtype, trtllmGenPrefillScale + dtype}
}

func withCache(in Inputs, dtype string, block int) Inputs {
	return withEngine(in, func(e *enginesettings.Settings) {
		e.CacheDType = dtype
		if block > 0 {
			e.BlockSize = block
		}
	})
}

func withTRTLLM(in Inputs, v bool) Inputs {
	return withEngine(in, func(e *enginesettings.Settings) { e.UseTRTLLMAttention = flag(v) })
}

// movedBy reports whether scaling any of the coefficients by 0.8 moves a prefill price of in.
func movedBy(t *testing.T, in Inputs, names []string) bool {
	t.Helper()
	want, _ := prefillAndDecode(mustKernel(t, in))
	for _, n := range names {
		if got, _ := prefillAndDecode(mustKernel(t, scaleEntry(in, n, 0.8))); got != want {
			return true
		}
	}
	return false
}

// The lane is chosen by the implementation the caller reports for the prefill primitive, and by
// nothing the settings say: not the cache dtype, the block size or use_trtllm_attention. The bf16
// pair is valid for one build and the fp8 pair for another, so the reported build picks the pair.
// Each lane coefficient moves the prices that read it and no others (the perturbation test of the
// layer).
func TestTRTLLMGenLaneIsChosenByTheReportedImplementation(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range []string{"b200", "b300", "gb200-nvl72", "gb300"} {
		t.Run(chip, func(t *testing.T) {
			raw := laneCase(t, d, chip)
			pooled := pricesOf(mustKernel(t, raw))
			bf16, fp8 := laneNames("bf16"), laneNames("fp8")

			onBF16 := keyedLane(raw, laneBF16)
			if pricesOf(mustKernel(t, onBF16)) == pooled {
				t.Error("the reported bf16 build runs and the lane is not read")
			}
			if !movedBy(t, onBF16, bf16) || movedBy(t, onBF16, fp8) {
				t.Error("the bf16 build must read the bf16 pair and not the fp8 one")
			}
			onFP8 := keyedLane(raw, laneFP8)
			if !movedBy(t, onFP8, fp8) || movedBy(t, onFP8, bf16) {
				t.Error("the fp8 build must read the fp8 pair and not the bf16 one")
			}
			// Nothing the settings say moves the choice: with an fp8 cache, a block size of 128 or
			// a stated true, the bf16 build still reads the bf16 pair and not the fp8 one. (The
			// cache dtype changes other prices, so the lane is detected by its coefficients.)
			for name, in := range map[string]Inputs{
				"an fp8 cache":     withCache(onBF16, "fp8", 0),
				"a block size 128": withCache(onBF16, "", 128),
				"a stated true":    withTRTLLM(onBF16, true),
			} {
				if !movedBy(t, in, bf16) || movedBy(t, in, fp8) {
					t.Errorf("%s: the choice of pair follows the setting, not the reported implementation", name)
				}
			}
			up, ud := prefillAndDecode(mustKernel(t, raw))
			sp, sd := prefillAndDecode(mustKernel(t, onBF16))
			if up == sp || ud != sd {
				t.Errorf("the lane moves prefill %v->%v and decode %v->%v; it prices prefill only", up, sp, ud, sd)
			}
		})
	}
}

// Latent attention is another kernel and keeps the pooled pair, so a stated true changes nothing
// on a model with no non-latent layer. The kernel does not infer from the settings that
// decode-context parallelism makes the engine ignore the setting: with a non-latent layer a
// stated true it cannot match to the reported implementation is refused.
func TestTRTLLMAttentionTrueOnLatentOnlyModelsChangesNothing(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	mla := quantCase(t, d, "deepseek-v3", "b200")
	want := pricesOf(mustKernel(t, mla))
	if got := pricesOf(mustKernel(t, withTRTLLM(mla, true))); got != want {
		t.Error("a stated true moves a latent-only model")
	}
	if movedBy(t, withTRTLLM(keyedLane(mla, laneBF16), true), append(laneNames("bf16"), laneNames("fp8")...)) {
		t.Error("a stated true reads the lane on a latent-only model")
	}
	if got := pricesOf(mustKernel(t, withTRTLLM(mla, false))); got != want {
		t.Error("a stated false moves a latent-only model")
	}
	var dcp Inputs
	for _, c := range perturbationScenarios(t, d, "codellama-34b-instruct-hf") {
		if strings.HasPrefix(c.label, "codellama-34b-instruct-hf/b200/") && strings.Contains(c.label, "/dcp2/") &&
			!strings.Contains(c.label, "/trtllm") {
			dcp = c.in
		}
	}
	if dcp.Model == nil {
		t.Fatal("no codellama-34b-instruct-hf scenario with decode-context parallelism 2 builds on b200")
	}
	if _, err := New(withTRTLLM(dcp, true)); err == nil || !strings.Contains(err.Error(), "attention.prefill") {
		t.Errorf("a stated true under decode-context parallelism with no reported implementation: %v", err)
	}
}

// A stated true with the lane removed from the sets, or whose pairs list another build, is refused
// rather than priced on the pooled pair.
func TestTRTLLMAttentionTrueRefusesWhenTheSetsCarryNoLane(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	base := keyedLane(laneCase(t, d, "b200"), laneBF16)
	strip := func(in Inputs, names ...string) Inputs {
		var sets []*coefficient.Set
		for _, s := range in.Coefficients {
			c := *s
			c.Coefficients = nil
			for _, e := range s.Coefficients {
				if !slices.Contains(names, e.Name) {
					c.Coefficients = append(c.Coefficients, e)
				}
			}
			sets = append(sets, &c)
		}
		in.Coefficients = sets
		return in
	}
	on := withTRTLLM(base, true)
	all := append(laneNames("bf16"), laneNames("fp8")...)
	_, err = New(strip(on, laneNames("bf16")...))
	if err == nil {
		t.Fatal("a stated true priced with no pair valid for the reported build")
	}
	for _, want := range []string{"use_trtllm_attention", "trtllm-gen", "b200", trtllmGenPrefillFloor + "fp8"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not name %q: %v", want, err)
		}
	}
	_, err = New(strip(on, all...))
	if err == nil || !strings.Contains(err.Error(), "no trtllm-gen prefill pair") {
		t.Errorf("a stated true with no lane in the sets: %v", err)
	}
	// The lane absent from the sets leaves an unstated deployment on the pooled pair, silently.
	k, err := New(strip(base, all...))
	if err != nil {
		t.Errorf("an unstated deployment is refused without the lane: %v", err)
	} else if why, ok := laneAssumed(k); ok {
		t.Errorf("an assumption is recorded for a lane the sets do not carry: %s", why)
	}
}

// Known-bad inputs. A part with no measurement of the kernel refuses a stated true; a stated
// false is refused everywhere; each unpriced setting is refused with its field and the
// measurement it needs.
func TestKernelFacingSettingsAreRefusedNotPricedAsTheDefault(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	b200 := laneCase(t, d, "b200")
	h200 := laneCase(t, d, "h200")
	cases := []struct {
		name string
		in   Inputs
		set  func(*enginesettings.Settings)
		want []string
	}{
		{"true on a Hopper part", h200, func(e *enginesettings.Settings) { e.UseTRTLLMAttention = flag(true) },
			[]string{"use_trtllm_attention", "h200", "trtllm-gen"}},
		{"false on a Blackwell part", b200, func(e *enginesettings.Settings) { e.UseTRTLLMAttention = flag(false) },
			[]string{"use_trtllm_attention", "false", "blis-registry#72", "Needed"}},
		{"an attention backend", b200, func(e *enginesettings.Settings) { e.AttentionBackend = "FLASHINFER" },
			[]string{"attention_config.backend", "FLASHINFER", "blis-registry#72", "Needed"}},
		{"an MLA prefill backend", b200, func(e *enginesettings.Settings) { e.MLAPrefillBackend = "TRTLLM_RAGGED" },
			[]string{"mla_prefill_backend", "TRTLLM_RAGGED", "blis-registry#67", "attention-only MLA prefill"}},
		{"a linear backend, as the engine spells it", b200, func(e *enginesettings.Settings) { e.LinearBackend = "FlashInfer-CUTLASS" },
			[]string{"linear_backend", "flashinfer_cutlass", "blis-registry#66", "FlashInfer CUTLASS"}},
		{"custom ops", b200, func(e *enginesettings.Settings) { e.CustomOps = []string{"+rms_norm", "all"} },
			[]string{"custom_ops", "+rms_norm,all", "Needed"}},
		{"decoder replay off", b200, func(e *enginesettings.Settings) { e.CUDAGraphDecoderReplay = flag(false) },
			[]string{"cudagraph_decoder_replay", "Needed"}},
		{"decoder replay capture sizes", b200, func(e *enginesettings.Settings) { e.DecoderReplayCUDAGraphCaptureSizes = []int{128, 256} },
			[]string{"decoder_replay_cudagraph_capture_sizes", "[128 256]", "Needed"}},
		{"an MSA decode backend", b200, func(e *enginesettings.Settings) { e.MiniMaxM3MSADecodeBackend = "flashinfer" },
			[]string{"minimax_m3_msa_decode_backend", "flashinfer", "blis-registry#72"}},
		{"a per-kind attention backend", b200, func(e *enginesettings.Settings) {
			e.AttentionBackendPerKind = map[string]string{"mla_attention": "FLASHINFER_MLA"}
		}, []string{"backend_per_kind", "FLASHINFER_MLA", "blis-registry#72"}},
		{"a per-quantization linear backend", b200, func(e *enginesettings.Settings) {
			e.LinearBackendPerQuant = map[string]string{"nvfp4_w4a16": "humming"}
		}, []string{"linear_backend_per_quant", "humming", "blis-registry#66"}},
		{"a drafter cache dtype", b200, func(e *enginesettings.Settings) {
			e.Speculative = &enginesettings.Speculative{Method: "eagle3", NumSpecTokens: 3, KVCacheDType: "fp8"}
		}, []string{"speculative_config.kv_cache_dtype", "fp8", "Needed"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := New(withEngine(c.in, c.set))
			if err == nil {
				t.Fatal("the kernel priced a setting it has no measurement for, as the default")
			}
			for _, w := range c.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("the refusal does not name %q: %v", w, err)
				}
			}
		})
	}
}

// Stating the engine's own default is not stating a choice: the values below ask for the kernel
// the engine picks anyway, so they price exactly as an unstated setting does.
func TestStatingTheEnginesDefaultPricesAsUnstated(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	base := laneCase(t, d, "b200")
	want := pricesOf(mustKernel(t, base))
	for name, f := range map[string]func(*enginesettings.Settings){
		"linear_backend auto":       func(e *enginesettings.Settings) { e.LinearBackend = "AUTO" },
		"attention backend auto":    func(e *enginesettings.Settings) { e.AttentionBackend = "auto" },
		"decoder replay on":         func(e *enginesettings.Settings) { e.CUDAGraphDecoderReplay = flag(true) },
		"msa decode backend triton": func(e *enginesettings.Settings) { e.MiniMaxM3MSADecodeBackend = "triton" },
		"no custom ops, no replay sizes": func(e *enginesettings.Settings) {
			e.CustomOps, e.DecoderReplayCUDAGraphCaptureSizes = []string{}, []int{}
		},
	} {
		got := pricesOf(mustKernel(t, withEngine(base, f)))
		if got != want {
			t.Errorf("%s prices differently from an unstated setting", name)
		}
	}
}

// Every unpriced setting says what would price it, and every refusal the table produces names
// the field the engine spells.
func TestEveryUnpricedSettingNamesItsMeasurement(t *testing.T) {
	seen := map[string]bool{}
	for _, u := range unpricedSettings {
		if u.Field == "" || len(strings.Fields(u.Needs)) < 12 {
			t.Errorf("%q states no measurement that would price it: %q", u.Field, u.Needs)
		}
		if seen[u.Field] {
			t.Errorf("%s is listed twice", u.Field)
		}
		seen[u.Field] = true
		if _, stated := u.Stated(enginesettings.Settings{}); stated {
			t.Errorf("%s reads an unset setting as stated", u.Field)
		}
	}
}

// refusalIssues is the open issue each refused setting belongs to, written out so a swap of two
// issues fails (X28). The titles are those of the issues: #66 linear_backend, #67 MLA prefill
// backends, #68 the drafter's KV cache dtype, #69 decoder-replay graphs, #71 custom_ops, #72
// per-backend attention (also use_trtllm_attention false). #70, the environment variables, reaches
// the kernel through no field and is named on the page.
var refusalIssues = map[string]string{
	"attention_config.backend":                                  "inference-sim/blis-registry#72",
	"attention_config.mla_prefill_backend":                      "inference-sim/blis-registry#67",
	"kernel_config.linear_backend":                              "inference-sim/blis-registry#66",
	"compilation_config.custom_ops":                             "inference-sim/blis-registry#71",
	"compilation_config.cudagraph_decoder_replay":               "inference-sim/blis-registry#69",
	"compilation_config.decoder_replay_cudagraph_capture_sizes": "inference-sim/blis-registry#69",
	"speculative_config.kv_cache_dtype":                         "inference-sim/blis-registry#68",
	"attention_config.minimax_m3_msa_decode_backend":            "inference-sim/blis-registry#72",
	"attention_config.backend_per_kind":                         "inference-sim/blis-registry#72",
	"kernel_config.linear_backend_per_quant":                    "inference-sim/blis-registry#66",
}

// The refusals page lists exactly the settings the kernel refuses whatever their value, each with
// its own issue, so the page can neither miss one nor describe one the code no longer refuses
// (X26), and an issue cannot be moved to another setting (X28).
func TestRefusalsPageListsTheUnpricedSettings(t *testing.T) {
	text, err := os.ReadFile(filepath.Join("..", "docs", "kernel", "reference", "refusals.md"))
	if err != nil {
		t.Fatal(err)
	}
	rows := map[string]string{}
	for _, l := range strings.Split(string(text), "\n") {
		if strings.HasPrefix(l, "| `") {
			cell := strings.SplitN(strings.TrimPrefix(l, "| `"), "`", 2)[0]
			rows[cell] = l
		}
	}
	var code []string
	for _, u := range unpricedSettings {
		code = append(code, u.Field)
		want, ok := refusalIssues[u.Field]
		if !ok || u.Issue != want {
			t.Errorf("%s names %q in the code, and the test says %q", u.Field, u.Issue, want)
		}
		if !strings.HasSuffix(strings.TrimSpace(rows[u.Field]), want+" |") {
			t.Errorf("the refusals page row for %s does not end with %s: %q", u.Field, want, rows[u.Field])
		}
	}
	if len(refusalIssues) != len(unpricedSettings) {
		t.Errorf("the test maps %d settings and the code refuses %d", len(refusalIssues), len(unpricedSettings))
	}
	// The setting that is priced when true and refused when false, and the environment variables,
	// are not in the table; the page names their issues.
	if !strings.HasSuffix(strings.TrimSpace(rows["attention_config.use_trtllm_attention"]), useTRTLLMFalseIssue+" |") {
		t.Errorf("the page row for use_trtllm_attention (false) does not name %s", useTRTLLMFalseIssue)
	}
	for _, name := range []string{"VLLM_FLASHINFER_ALLREDUCE_BACKEND", "VLLM_MARLIN_USE_ATOMIC_ADD"} {
		if !strings.Contains(string(text), name) {
			t.Errorf("the page does not name %s", name)
		}
	}
	if !strings.Contains(string(text), "inference-sim/blis-registry#70") {
		t.Error("the page does not name the issue of the environment variables")
	}
	var page []string
	for cell := range rows {
		if cell != "attention_config.use_trtllm_attention" && strings.Contains(cell, "_") && strings.Contains(cell, "_config.") {
			page = append(page, cell)
		}
	}
	sort.Strings(code)
	sort.Strings(page)
	sameSet(t, "the unpriced setting", page, code)
}

// A model that mixes latent and plain layers prices only the plain ones on the lane: the MLA and
// sparse-MLA backends never read use_trtllm_attention.
func TestTheLaneIsForPlainLayersOnly(t *testing.T) {
	k := &Kernel{laneRuns: true, lanePair: prefillPair{20, 0.34}, attentionPrefillFloor: 16, attentionPrefillScale: 0.32}
	for _, kind := range []model.AttentionKind{model.AttentionMLA, model.AttentionSparseMLA} {
		if got := k.prefillFor(kind); got != (prefillPair{16, 0.32}) {
			t.Errorf("%s layers are priced on the lane: %v", kind, got)
		}
	}
	for _, kind := range []model.AttentionKind{model.AttentionGQA, model.AttentionSWA} {
		if got := k.prefillFor(kind); got != (prefillPair{20, 0.34}) {
			t.Errorf("%s layers are not priced on the lane: %v", kind, got)
		}
	}
}

// VAL-8: the lane is priced only when its pair lists the implementation of the primitive it
// prices. The known-bad inputs are the siblings (the same library and variant at another hash, the
// same hash under another variant, another library) and the mixed pool, where the entry is valid
// for a decode kernel A and prefill runs B. Each is refused through the real kernel.New.
func TestTRTLLMGenLaneIsPricedOnlyForTheImplementationOfThePrimitiveItPrices(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	base := withCache(laneCase(t, d, "b200"), "", 0)
	keyed := keyedLane(base, laneBF16)
	pooled := pricesOf(mustKernel(t, base))

	// The exact implementation: the lane prices, differently from the pooled pair, whether the
	// setting is unstated or true, and the selected entries report exact for the prefill primitive.
	for _, stated := range []bool{false, true} {
		in := keyed
		if stated {
			in = withTRTLLM(keyed, true)
		}
		k := mustKernel(t, in)
		if pricesOf(k) == pooled {
			t.Errorf("stated=%v: the exact implementation runs and the lane is not read", stated)
		}
		if why, ok := laneAssumed(k); ok {
			t.Errorf("stated=%v: the lane is priced and an assumption says it is not: %s", stated, why)
		}
		found := false
		for _, o := range k.origins {
			if o.Name == trtllmGenPrefillFloor+"bf16" {
				found = true
				if o.KernelMatch != string(coefficient.KernelExact) || o.Primitive != "attention.prefill" {
					t.Errorf("stated=%v: %s is selected as %q for %q, want exact for attention.prefill", stated, o.Name, o.KernelMatch, o.Primitive)
				}
			}
		}
		if !found {
			t.Errorf("stated=%v: the lane's floor is not among the selected entries", stated)
		}
	}

	siblingHash := plan.KernelImpl{Library: laneBF16.Library, Variant: laneBF16.Variant, CodeHash: "ffffffffffff"}
	siblingVariant := plan.KernelImpl{Library: laneBF16.Library, Variant: "fixture-native-prefill", CodeHash: laneBF16.CodeHash}
	otherLibrary := plan.KernelImpl{Library: "other-lib", Variant: laneBF16.Variant, CodeHash: laneBF16.CodeHash}
	cases := []struct {
		name    string
		running map[plan.Primitive][]plan.KernelImpl
	}{
		{"a sibling hash", map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionPrefill: {siblingHash}}},
		{"a sibling variant", map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionPrefill: {siblingVariant}}},
		{"another library", map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionPrefill: {otherLibrary}}},
		{"no implementation reported", nil},
		// The mixed pool: the pair is valid for A and the pool runs A for decode and B for prefill.
		{"A for decode and B for prefill", map[plan.Primitive][]plan.KernelImpl{
			plan.PrimAttentionDecode: {laneBF16}, plan.PrimAttentionPrefill: {siblingVariant}}},
		// A decode-only report: the pair prices prefill and the pool says nothing about it.
		{"A for decode only", map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionDecode: {laneBF16}}},
		// Two prefill kernels: one value cannot price both.
		{"A and B for prefill", map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionPrefill: {laneBF16, siblingVariant}}},
	}
	for _, c := range cases {
		in := keyed
		in.Implementations = c.running
		// A stated true is refused, naming the setting, the entry, its hash and what prefill runs.
		_, err := New(withTRTLLM(in, true))
		if err == nil {
			t.Errorf("%s: a stated true is priced on a pair valid for another implementation", c.name)
		} else {
			for _, want := range []string{"use_trtllm_attention", "VAL-8", trtllmGenPrefillFloor + "bf16", laneBF16.CodeHash} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("%s: the refusal does not name %q: %v", c.name, want, err)
				}
			}
		}
		// Unstated keeps the pooled pair and says why.
		k, err := New(in)
		if err != nil {
			t.Errorf("%s: an unstated deployment is refused: %v", c.name, err)
			continue
		}
		if got := pricesOf(k); got != pooled {
			t.Errorf("%s: an unstated deployment is not on the pooled pair", c.name)
		}
		if why, ok := laneAssumed(k); !ok || !strings.Contains(why, "pooled") {
			t.Errorf("%s: the unpriced lane is not recorded as an assumption: %q", c.name, why)
		}
	}

	// The regression of the review: the pairs are valid for a decode kernel A, and the pool runs A
	// for decode and a different kernel B for prefill. The pool-wide union holds A; the primitive
	// the pair prices does not.
	forA := keyLaneEntries(base, map[string]plan.KernelImpl{"bf16": laneDecode, "fp8": laneDecode})
	forA.Implementations = map[plan.Primitive][]plan.KernelImpl{
		plan.PrimAttentionDecode: {laneDecode}, plan.PrimAttentionPrefill: {laneBF16}}
	if _, err := New(withTRTLLM(forA, true)); err == nil || !strings.Contains(err.Error(), "attention.prefill runs") {
		t.Errorf("a pair valid for decode kernel A prices a prefill that runs B: %v", err)
	}
	kA := mustKernel(t, forA)
	if got := pricesOf(kA); got != pooled {
		t.Error("an unstated deployment whose pair is valid for the decode kernel only is not on the pooled pair")
	}

	// A stated false contradicts a reported build the lane lists, and the setting is not priced.
	if _, err := New(withTRTLLM(keyed, false)); err == nil || !strings.Contains(err.Error(), "contradicts") {
		t.Errorf("a stated false with the lane's build reported: %v", err)
	}
	// Both pairs listing the build leave the cache dtype unknown.
	both := keyLaneEntries(keyed, map[string]plan.KernelImpl{"bf16": laneBF16, "fp8": laneBF16})
	if _, err := New(both); err == nil || !strings.Contains(err.Error(), "both list") {
		t.Errorf("both pairs list the reported build: %v", err)
	}
}

// The registry's lane states no implementation, so it is not priced whatever the pool reports: a
// stated true is refused naming the hash it needs, and an unstated setting keeps the pooled pair.
func TestTheCommittedTRTLLMGenLaneIsNotPricedWithoutAKernelHash(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range []string{"b200", "b300", "gb200-nvl72", "gb300"} {
		base := withCache(laneCase(t, d, chip), "", 0)
		for _, in := range []Inputs{base, withPrefill(base, laneBF16), withPrefill(base, laneFP8)} {
			_, err := New(withTRTLLM(in, true))
			if err == nil || !strings.Contains(err.Error(), "kernel_impls") || !strings.Contains(err.Error(), "blis-registry#73") {
				t.Errorf("%s: a stated true on the unkeyed lane: %v", chip, err)
			}
			k := mustKernel(t, in)
			if _, ok := laneAssumed(k); !ok {
				t.Errorf("%s: the unpriced lane is not recorded", chip)
			}
			if movedBy(t, in, append(laneNames("bf16"), laneNames("fp8")...)) {
				t.Errorf("%s: an unkeyed lane entry moves a price", chip)
			}
		}
	}
}
