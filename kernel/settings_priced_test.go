package latencykernel

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
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

func flag(b bool) *bool { return &b }

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
		{"true", h200, func(e *enginesettings.Settings) { e.UseTRTLLMAttention = flag(true) },
			[]string{"use_trtllm_attention", "true", "blis-registry#72", "Needed"}},
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
