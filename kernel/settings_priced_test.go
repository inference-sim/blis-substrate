package latencykernel

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

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

// The lane is chosen by what v0.32.0's FlashInfer backend runs, not by the flag alone
// (vllm/utils/flashinfer.py:697-789, vllm/v1/attention/backends/flashinfer.py:1429-1433): an
// unstated setting with an auto cache, or a stated block size of 128, runs the trtllm-gen prefill
// and prices as a stated true does; with an fp8 cache and neither, the unstated price is the
// pooled pair and a stated true is the fp8 lane. Each lane coefficient moves the prices that read
// it and no others (the perturbation test of the layer).
func TestTRTLLMGenLaneIsChosenByWhatTheEngineRuns(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range []string{"b200", "b300", "gb200-nvl72", "gb300"} {
		t.Run(chip, func(t *testing.T) {
			base := laneCase(t, d, chip)
			bf16, fp8 := laneNames("bf16"), laneNames("fp8")

			// An auto cache: unstated runs the kernel, so stated true is the same price.
			auto := withCache(base, "", 0)
			if pricesOf(mustKernel(t, auto)) != pricesOf(mustKernel(t, withTRTLLM(auto, true))) {
				t.Error("an auto cache prices a stated true differently from an unstated setting: vLLM runs one kernel")
			}
			if !movedBy(t, auto, bf16) || movedBy(t, auto, fp8) {
				t.Error("an unstated auto-cache deployment must read the bf16 lane and not the fp8 one")
			}

			// An fp8 cache: unstated runs the native kernel (pooled pair); true runs trtllm-gen.
			f8 := withCache(base, "fp8", 0)
			up, ud := prefillAndDecode(mustKernel(t, f8))
			sp, sd := prefillAndDecode(mustKernel(t, withTRTLLM(f8, true)))
			if up == sp {
				t.Error("an fp8 cache prices a stated true as an unstated one: the lane is not read")
			}
			if ud != sd {
				t.Errorf("a stated true moved decode from %v to %v; v0.32.0 decodes with trtllm-gen already", ud, sd)
			}
			if movedBy(t, f8, append(bf16, fp8...)) {
				t.Error("an unstated fp8-cache deployment reads a lane coefficient")
			}
			if !movedBy(t, withTRTLLM(f8, true), fp8) || movedBy(t, withTRTLLM(f8, true), bf16) {
				t.Error("a stated true on an fp8 cache must read the fp8 lane and not the bf16 one")
			}

			// A stated block size of 128 forces the kernel, so unstated is the same price.
			big := withCache(base, "fp8", 128)
			if pricesOf(mustKernel(t, big)) != pricesOf(mustKernel(t, withTRTLLM(big, true))) {
				t.Error("a block size of 128 prices a stated true differently from an unstated setting")
			}
			if !movedBy(t, big, fp8) {
				t.Error("a block size of 128 does not read the lane")
			}
		})
	}
}

// Where vLLM ignores the setting, a stated true changes nothing: latent attention never reads it,
// and decode-context parallelism returns False for the prefill whatever it says.
func TestTRTLLMAttentionTrueChangesNothingWhereVLLMIgnoresIt(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	mla := quantCase(t, d, "deepseek-v3", "b200")
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
	for name, in := range map[string]Inputs{"mla": mla, "dcp": dcp} {
		want := pricesOf(mustKernel(t, in))
		if got := pricesOf(mustKernel(t, withTRTLLM(in, true))); got != want {
			t.Errorf("%s: a stated true moves the price", name)
		}
		if movedBy(t, withTRTLLM(in, true), append(laneNames("bf16"), laneNames("fp8")...)) {
			t.Errorf("%s: a stated true reads the lane", name)
		}
	}
	// Latent-only: a stated false is a no-op too, while a model with a plain layer refuses it.
	if got := pricesOf(mustKernel(t, withTRTLLM(mla, false))); got != pricesOf(mustKernel(t, mla)) {
		t.Error("a stated false moves a latent-only model")
	}
}

// A stated true on a cache dtype the sweep does not hold, or with the lane removed from the sets,
// is refused rather than priced on the pooled pair.
func TestTRTLLMAttentionTrueRefusesWhenTheSetsCarryNoLane(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	base := laneCase(t, d, "b200")
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
	on := withTRTLLM(withCache(base, "", 0), true)
	_, err = New(strip(on, laneNames("bf16")...))
	if err == nil {
		t.Fatal("a stated true priced with no lane entry in the sets")
	}
	for _, want := range []string{"use_trtllm_attention", "trtllm-gen", "b200", trtllmGenPrefillFloor + "bf16"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not name %q: %v", want, err)
		}
	}
	if _, err := New(withTRTLLM(withCache(base, "nvfp4", 0), true)); err == nil || !strings.Contains(err.Error(), "nvfp4") {
		t.Errorf("a stated true at an nvfp4 cache: %v", err)
	}
	// The lane absent from the sets leaves an unstated deployment on the pooled pair.
	if _, err := New(strip(withCache(base, "", 0), append(laneNames("bf16"), laneNames("fp8")...)...)); err != nil {
		t.Errorf("an unstated deployment is refused without the lane: %v", err)
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
