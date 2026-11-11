package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
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
		{"modelopt_mixed", bf16, "modelopt_mixed", []string{"modelopt_mixed", "MIXED_PRECISION", "quantized_layers", "Needed", "ServedWeights"}},
		{"modelopt_mixed on an fp8 checkpoint", mxfp8, "modelopt_mixed", []string{"MIXED_PRECISION"}},
		// The spelling is vLLM's, so the cost model does not recognise it on any graph, the
		// checkpoint's own included; the engine model states ServedWeights instead.
		{"modelopt_mixed on its own checkpoint", quantCase(t, d, "minimax-m3-nvfp4", "b200"), "modelopt_mixed", []string{"MIXED_PRECISION", "not recognised", "ServedWeights"}},
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

// mixedPrecisionServed is the layout the engine model states for nvidia/MiniMax-M3-NVFP4: MXFP8
// linears (one scale byte per 32 weights), NVFP4 routed experts, an FP32 router, a BF16 head and
// a BF16 embedding.
var mixedPrecisionServed = enginesettings.ServedWeights{Linear: "fp8", LinearScaleElements: 32,
	RoutedExperts: "nvfp4", Router: "fp32", Head: "bf16", Embedding: "bf16"}

// The engine model states the layout it serves and the kernel proves the graph stores exactly
// it. The real MiniMax-M3 NVFP4 graph matches; each one-edit variant of it, and each other
// catalog layout, does not, and is refused with the class named. The deletions and the wrong
// dtypes are the counterexamples a predicate that checks only what happens to be present
// would accept.
func TestTheServedLayoutIsProvedAgainstTheGraph(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	served := func(in Inputs, sw *enginesettings.ServedWeights) Inputs {
		return withEngine(in, func(e *enginesettings.Settings) { e.ServedWeights = sw })
	}
	base := quantCase(t, d, "minimax-m3-nvfp4", "b200")
	if _, err := New(served(base, &mixedPrecisionServed)); err != nil {
		t.Fatalf("the checkpoint's own layout was refused: %v", err)
	}
	// Nil is the graph as stored, as before.
	if _, err := New(base); err != nil {
		t.Fatal(err)
	}
	edit := func(f func(g *model.Graph)) Inputs {
		in := base
		g, err := in.Model.ForCapability(10)
		if err != nil {
			t.Fatal(err)
		}
		c := *g
		c.LayerKinds = append([]model.LayerKind(nil), g.LayerKinds...)
		for i := range c.LayerKinds {
			c.LayerKinds[i].Nodes = append([]model.Node(nil), g.LayerKinds[i].Nodes...)
		}
		c.Head = append([]model.Node(nil), g.Head...)
		c.Embedding = append([]model.Node(nil), g.Embedding...)
		f(&c)
		in.Model = &c
		return served(in, &mixedPrecisionServed)
	}
	dropNode := func(g *model.Graph, match func(n model.Node) bool) {
		for i := range g.LayerKinds {
			kept := g.LayerKinds[i].Nodes[:0]
			for _, n := range g.LayerKinds[i].Nodes {
				if !match(n) {
					kept = append(kept, n)
				}
			}
			g.LayerKinds[i].Nodes = kept
			g.LayerKinds[i].Edges = nil
		}
	}
	setNode := func(g *model.Graph, match func(n model.Node) bool, dt model.DType) {
		for i := range g.LayerKinds {
			for j := range g.LayerKinds[i].Nodes {
				if match(g.LayerKinds[i].Nodes[j]) {
					g.LayerKinds[i].Nodes[j].WeightDType = dt
				}
			}
		}
	}
	isRouter := func(n model.Node) bool { return n.Op == model.OpGEMM && n.N == 128 && n.Role == "router" }
	isRouted := func(n model.Node) bool { return n.Op == model.OpGroupedGEMM && !n.IsSharedExpert() }
	isAttn := func(n model.Node) bool { return n.Op == model.OpGEMM && n.Role == "qkv_proj" }
	for name, c := range map[string]struct {
		in   Inputs
		want string
	}{
		"router removed": {edit(func(g *model.Graph) { dropNode(g, isRouter) }), "router"},
		"router at bf16": {edit(func(g *model.Graph) { setNode(g, isRouter, model.DTypeBF16) }), "router"},
		"head removed": {edit(func(g *model.Graph) {
			kept := g.Head[:0]
			for _, n := range g.Head {
				if n.Role != "lm_head" {
					kept = append(kept, n)
				}
			}
			g.Head = kept
		}), "head"},
		"head at fp8": {edit(func(g *model.Graph) {
			for i := range g.Head {
				if g.Head[i].Role == "lm_head" {
					g.Head[i].WeightDType = model.DTypeFP8
				}
			}
		}), "head"},
		"embedding removed": {edit(func(g *model.Graph) { g.Embedding = nil }), "embedding"},
		"embedding at fp8": {edit(func(g *model.Graph) {
			for i := range g.Embedding {
				if g.Embedding[i].Op == model.OpGather {
					g.Embedding[i].RowBytes /= 2
				}
			}
		}), "embedding"},
		"routed experts at fp8": {edit(func(g *model.Graph) { setNode(g, isRouted, model.DTypeFP8) }), "routed_experts"},
		"attention at bf16":     {edit(func(g *model.Graph) { setNode(g, isAttn, model.DTypeBF16) }), "linear"},
		"no microscaling scale": {edit(func(g *model.Graph) { g.Global.WeightScale = nil }), "linear scale"},
		"shared expert in the NVFP4 node": {edit(func(g *model.Graph) {
			for i := range g.LayerKinds {
				for j := range g.LayerKinds[i].Nodes {
					if isRouted(g.LayerKinds[i].Nodes[j]) {
						g.LayerKinds[i].Nodes[j].SharedExperts = 1
					}
				}
			}
		}), "linear"},
		"the MXFP8 checkpoint": {served(quantCase(t, d, "minimax-m3-mxfp8", "b200"), &mixedPrecisionServed), "routed_experts"},
		"the bf16 checkpoint":  {served(quantCase(t, d, "minimax-m3", "b200"), &mixedPrecisionServed), "linear"},
		"gpt-oss":              {served(quantCase(t, d, "gpt-oss-120b", "b200"), &mixedPrecisionServed), "linear"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := New(c.in)
			if err == nil {
				t.Fatal("a graph that is not the served layout was priced")
			}
			for _, w := range []string{"served_weights_mismatch", c.want} {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("the refusal does not name %q: %v", w, err)
				}
			}
		})
	}
}

// The graph stores the layout whose quantized checkpoint it describes at a lower weight
// footprint, and prices differently from the MXFP8 checkpoint's.
func TestTheMixedPrecisionGraphPricesNVFP4ExpertsUnderFP8Linears(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range []string{"b200", "b300"} {
		mx := mustKernel(t, quantCase(t, d, "minimax-m3-mxfp8", chip))
		nv := mustKernel(t, quantCase(t, d, "minimax-m3-nvfp4", chip))
		if mx.FixedBytes().Weights <= nv.FixedBytes().Weights {
			t.Errorf("%s: NVFP4 experts hold %d weight bytes a rank, MXFP8 %d; the experts are most of the checkpoint",
				chip, nv.FixedBytes().Weights, mx.FixedBytes().Weights)
		}
		if pricesOf(mx) == pricesOf(nv) {
			t.Errorf("%s: the NVFP4 graph prices as the MXFP8 one", chip)
		}
	}
}
