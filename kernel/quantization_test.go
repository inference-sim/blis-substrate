package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/load"
)

// A stated quantization that names the checkpoint's own method prices exactly as an unstated one;
// one that does not is refused with the mismatch named, never priced silently (see quantization.go).

func withQuant(in Inputs, q string) Inputs {
	return withEngine(in, func(e *enginesettings.Settings) { e.Quantization = q })
}

func TestStatingTheCheckpointsOwnQuantizationPricesAsUnstated(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		model  string
		stated []string
	}{
		{"minimax-m3-mxfp8", []string{"mxfp8"}},
		{"gpt-oss-120b", []string{"gpt_oss_mxfp4", "mxfp4"}},
	} {
		for _, chip := range []string{"b200", "b300"} {
			base := quantCase(t, d, c.model, chip)
			want := pricesOf(mustKernel(t, base))
			for _, q := range c.stated {
				k := mustKernel(t, withQuant(base, q))
				if got := pricesOf(k); got != want {
					t.Errorf("%s on %s: a stated %q prices differently from an unstated quantization", c.model, chip, q)
				}
				// Only a spelling vLLM rewrites is recorded as an override.
				rewritten := q == "mxfp4"
				if got := len(k.layout.Overrides) > len(mustKernel(t, base).layout.Overrides); got != rewritten {
					t.Errorf("%s: stated %q records an override: %v, want %v", c.model, q, got, rewritten)
				}
			}
		}
	}
}

func TestAStatedQuantizationTheCheckpointDoesNotDeclareIsRefused(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	bf16 := quantCase(t, d, "minimax-m3", "b200")
	mxfp8 := quantCase(t, d, "minimax-m3-mxfp8", "b200")
	oss := quantCase(t, d, "gpt-oss-120b", "b200")
	for _, c := range []struct {
		name string
		in   Inputs
		q    string
		want []string
	}{
		{"mxfp8 on a bf16 checkpoint", bf16, "mxfp8", []string{"mxfp8", "does not store MXFP8", "online"}},
		{"mxfp8 on an fp8 checkpoint with other scales", quantCase(t, d, "minimax-m2.5", "b200"), "mxfp8", []string{"does not store MXFP8", "one scale per 32"}},
		{"mxfp8 on gpt-oss", oss, "mxfp8", []string{"mxfp8", "does not store MXFP8"}},
		{"gpt_oss_mxfp4 on an fp8 checkpoint", mxfp8, "gpt_oss_mxfp4", []string{"gpt_oss_mxfp4", "not gpt-oss", "MiniMaxM3"}},
		{"gpt_oss_mxfp4 on a bf16 checkpoint", bf16, "gpt_oss_mxfp4", []string{"not gpt-oss"}},
		{"modelopt_mxfp8", mxfp8, "modelopt_mxfp8", []string{"modelopt_mxfp8", "quant_algo MXFP8"}},
		{"modelopt_mixed", bf16, "modelopt_mixed", []string{"modelopt_mixed", "MIXED_PRECISION", "quantized_layers", "Needed"}},
		{"modelopt_mixed on an fp8 checkpoint", mxfp8, "modelopt_mixed", []string{"MIXED_PRECISION"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := New(withQuant(c.in, c.q))
			if err == nil {
				t.Fatal("the kernel priced a quantization the checkpoint does not declare")
			}
			for _, w := range c.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("the refusal does not name %q: %v", w, err)
				}
			}
		})
	}
}

// Before this change a stated mxfp4 on gpt-oss priced every weight at MXFP4, attention included.
// A stated mxfp4 anywhere else is now refused too: vLLM refuses it beside a checkpoint that
// declares another method, and quantizes a bf16 one online, and neither is priced.
func TestMxfp4OnAGraphThatIsNotGptOssIsRefused(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"minimax-m3", "kimi-k2.5", "minimax-m3-mxfp8"} {
		_, err := New(withQuant(quantCase(t, d, name, "b200"), "mxfp4"))
		if err == nil || !strings.Contains(err.Error(), "not gpt-oss") {
			t.Errorf("%s with a stated mxfp4: %v", name, err)
		}
	}
}

// A DeepSeek-V4 checkpoint declares fp8 with 32 x 32 blocks, and vLLM's DeepseekV4FP8Config claims
// its model_type whatever was stated, so a stated mxfp8 is not a mismatch there: it prices as
// unstated and records the override, and never reaches the MXFP8 rule.
func TestMxfp8StatedOnADeepSeekV4CheckpointIsOverriddenNotAccepted(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	base := quantCase(t, d, "deepseek-v4.1-flash", "b200")
	k := mustKernel(t, withQuant(base, "mxfp8"))
	if pricesOf(k) != pricesOf(mustKernel(t, base)) {
		t.Error("a stated mxfp8 moves a DeepSeek-V4 price")
	}
	found := false
	for _, o := range k.layout.Overrides {
		found = found || (o.Field == "quantization" && o.Requested == "mxfp8" && o.Resolved == "deepseek_v4_fp8")
	}
	if !found {
		t.Error("the override to deepseek_v4_fp8 is not recorded")
	}
}
