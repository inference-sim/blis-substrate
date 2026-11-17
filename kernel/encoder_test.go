package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
	"gopkg.in/yaml.v3"
)

// towerOf builds a Kimi-MoonViT-shaped tower whose output is hidden wide.
func towerOf(hidden int) model.Tower {
	return model.Tower{
		Modality: model.TowerImage, Input: model.TowerInput{PatchSize: 14, Channels: 3},
		MergeRatio: 4, OutputHiddenSize: hidden, Parallel: model.EncoderTensor, WeightDType: model.DTypeBF16,
		Embedding: []model.EncoderNode{
			{Op: model.EncoderGEMM, Role: "patch_embed", N: 1152, K: 588, Bias: true, TP: model.TPReplicated},
			{Op: model.EncoderElementwise, Role: "pos_embed", BytesPerRow: 4608, Params: 64 * 64 * 1152},
		},
		LayerKinds: []model.EncoderLayerKind{{ID: "block", Nodes: []model.EncoderNode{
			{Op: model.EncoderGEMM, Role: "qkv", N: 3456, K: 1152, Bias: true, TP: model.TPColumn},
			{Op: model.EncoderAttention, NumHeads: 16, HeadDim: 72, TP: model.TPSharded},
			{Op: model.EncoderGEMM, Role: "o", N: 1152, K: 1152, Bias: true, TP: model.TPRow},
			{Op: model.EncoderGEMM, Role: "fc1", N: 4304, K: 1152, Bias: true, TP: model.TPColumn},
			{Op: model.EncoderGEMM, Role: "fc2", N: 1152, K: 4304, Bias: true, TP: model.TPRow},
		}}},
		Stack: model.Stack{Pattern: []string{"block"}, Repeat: 27},
		Projector: []model.EncoderNode{
			{Op: model.EncoderGEMM, Role: "merge_fc1", N: 4608, K: 4608, Bias: true, Rows: model.RowsOutputTokens},
			{Op: model.EncoderGEMM, Role: "merge_fc2", N: hidden, K: 4608, Bias: true, Rows: model.RowsOutputTokens},
		},
	}
}

// towerCase opens a catalog text model at tensor-parallel width tp with a vision tower grafted
// on, and the engine's statement of what it does with it.
func towerCase(t *testing.T, tp int, tower func(int) model.Tower, run *plan.EncoderRun) Inputs {
	t.Helper()
	return towerCaseOn(t, "", tp, tower, run)
}

// towerCaseOn is towerCase on the named chip ("" for the first that builds).
func towerCaseOn(t *testing.T, chip string, tp int, tower func(int) model.Tower, run *plan.EncoderRun) Inputs {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range perturbationScenarios(t, d, "qwen3-30b-a3b") {
		p := c.in.Deployment.Pools[c.in.PoolIndex].Parallel
		if p.TP != tp || p.DP > 1 || p.PP > 1 || (chip != "" && c.in.Chip.Name != chip) {
			continue
		}
		b, err := yaml.Marshal(*c.in.Model)
		if err != nil {
			t.Fatal(err)
		}
		var g model.Graph
		if err := yaml.Unmarshal(b, &g); err != nil {
			t.Fatal(err)
		}
		g.Modality = model.ModalityMultimodal
		g.Encoder = &model.Encoder{Towers: []model.Tower{tower(g.Global.HiddenSize)}}
		in := c.in
		in.Model = &g
		lw := *in.Lowered
		lw.Encoder = run
		in.Lowered = &lw
		return in
	}
	t.Fatalf("no tensor-parallel width %d case of the model builds", tp)
	return Inputs{}
}

func build(t *testing.T, in Inputs) *Kernel {
	t.Helper()
	k, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

var sharded = &plan.EncoderRun{Resident: true, Mode: plan.EncoderWeightsSharded}

func img(h, w int) EncodeImage { return EncodeImage{HeightPx: h, WidthPx: w, Frames: 1} }

func TestEncodeTimeRisesWithTheImages(t *testing.T) {
	k := build(t, towerCase(t, 4, towerOf, sharded))
	none, err := k.EncodeTime(EncodeBatch{})
	if err != nil || none.Total != 0 {
		t.Fatalf("no images should cost nothing: %v, %v", none.Total, err)
	}
	small, _ := k.EncodeTime(EncodeBatch{Images: []EncodeImage{img(224, 224)}})
	large, _ := k.EncodeTime(EncodeBatch{Images: []EncodeImage{img(896, 896)}})
	two, _ := k.EncodeTime(EncodeBatch{Images: []EncodeImage{img(224, 224), img(224, 224)}})
	if small.Total <= 0 || large.Total <= small.Total {
		t.Errorf("a larger image should take longer: %v vs %v", large.Total, small.Total)
	}
	if two.Total <= small.Total {
		t.Errorf("two images should take longer than one: %v vs %v", two.Total, small.Total)
	}
	if small.OutputTokens[0] != 8*8 {
		t.Errorf("a 224px image under a 2x2 merge of 14px patches yields 64 tokens, got %d", small.OutputTokens[0])
	}
	if small.HostPerImage != 0 || small.HostPerImageNote == "" {
		t.Errorf("the per-image host cost is unmeasured and must say so: %v %q", small.HostPerImage, small.HostPerImageNote)
	}
}

func TestEncodeTimePerImageSumsToTheBatch(t *testing.T) {
	k := build(t, towerCase(t, 4, towerOf, sharded))
	e, err := k.EncodeTime(EncodeBatch{Images: []EncodeImage{img(224, 224), img(896, 448), img(448, 448)}})
	if err != nil {
		t.Fatal(err)
	}
	var sum int64
	for _, d := range e.PerImage {
		sum += int64(d)
	}
	if len(e.PerImage) != 3 || sum != int64(e.Total) {
		t.Errorf("per-image %v should sum to the total %v", e.PerImage, e.Total)
	}
	if e.PerImage[1] <= e.PerImage[2] || e.PerImage[2] <= e.PerImage[0] {
		t.Errorf("a bigger image should take a bigger share: %v", e.PerImage)
	}
}

// An image is attended on its own: two images of R rows cost less attention than one of 2R.
func TestTowerAttentionDoesNotCrossImages(t *testing.T) {
	attnOnly := func(h int) model.Tower {
		tw := towerOf(h)
		tw.Embedding, tw.Projector = nil, nil
		tw.LayerKinds = []model.EncoderLayerKind{{ID: "block", Nodes: []model.EncoderNode{
			{Op: model.EncoderAttention, NumHeads: 16, HeadDim: 72, TP: model.TPSharded}}}}
		return tw
	}
	k := build(t, towerCase(t, 4, attnOnly, sharded))
	one, _ := k.EncodeTime(EncodeBatch{Images: []EncodeImage{img(896, 896)}})
	two, _ := k.EncodeTime(EncodeBatch{Images: []EncodeImage{img(896, 448), img(896, 448)}})
	if two.Total >= one.Total {
		t.Errorf("two half-size images (%v) should attend less than one whole (%v)", two.Total, one.Total)
	}
}

func TestEncodeIsRefusedByNameWhereNothingIsPriced(t *testing.T) {
	// The engine serves the decoder only (language_model_only): an image cannot be encoded.
	k := build(t, towerCase(t, 4, towerOf, &plan.EncoderRun{Resident: false}))
	if _, err := k.EncodeTime(EncodeBatch{Images: []EncodeImage{img(224, 224)}}); err == nil ||
		!strings.Contains(err.Error(), "without its vision tower") {
		t.Errorf("a decoder-only deployment should refuse an image by name, got %v", err)
	}
	// The engine states nothing about the tower.
	k = build(t, towerCase(t, 4, towerOf, nil))
	if _, err := k.EncodeTime(EncodeBatch{Images: []EncodeImage{img(224, 224)}}); err == nil ||
		!strings.Contains(err.Error(), "states no encoder run") {
		t.Errorf("an unstated engine should be refused, got %v", err)
	}
	// A graph with no tower at all.
	in := towerCase(t, 4, towerOf, sharded)
	g := *in.Model
	g.Encoder, g.Modality = nil, ""
	in.Model = &g
	k = build(t, in)
	if _, err := k.EncodeTime(EncodeBatch{Images: []EncodeImage{img(224, 224)}}); err == nil ||
		!strings.Contains(err.Error(), "prices no vision tower") {
		t.Errorf("a text-only graph should refuse an image, got %v", err)
	}
}

// The tower is resident exactly when the engine keeps it, and divides as the mode says.
func TestResidentWeightsIncludeTheTowerUnlessTheEngineDropsIt(t *testing.T) {
	const tp = 4
	base := build(t, towerCase(t, tp, towerOf, &plan.EncoderRun{Resident: false})).FixedBytes().Weights
	tower := towerOf(1)
	var shardedBytes, wholeBytes float64
	add := func(n model.EncoderNode) {
		b := float64(n.ParamCount()) * 2
		wholeBytes += b
		switch n.TP {
		case model.TPColumn, model.TPRow, model.TPSharded:
			shardedBytes += b / tp
		default:
			shardedBytes += b
		}
	}
	for _, n := range tower.Embedding {
		add(n)
	}
	for i := 0; i < 27; i++ {
		for _, n := range tower.LayerKinds[0].Nodes {
			add(n)
		}
	}
	// The projector's last GEMM is hidden wide; take the graph's own width.
	in := towerCase(t, tp, towerOf, sharded)
	hidden := in.Model.Global.HiddenSize
	for _, n := range towerOf(hidden).Projector {
		add(n)
	}

	got := build(t, in).FixedBytes().Weights - base
	if d := float64(got) - shardedBytes; d < -2 || d > 2 {
		t.Errorf("weights-sharded tower adds %d bytes, want %.0f", got, shardedBytes)
	}
	split := &plan.EncoderRun{Resident: true, Mode: plan.EncoderBatchSplit}
	got = build(t, towerCase(t, tp, towerOf, split)).FixedBytes().Weights - base
	if d := float64(got) - wholeBytes; d < -2 || d > 2 {
		t.Errorf("batch-split tower adds %d bytes, want the whole %.0f", got, wholeBytes)
	}
	// A tower the model code replicates is whole on every rank whatever the mode.
	repl := func(h int) model.Tower { tw := towerOf(h); tw.Parallel = model.EncoderReplicated; return tw }
	got = build(t, towerCase(t, tp, repl, sharded)).FixedBytes().Weights - base
	if d := float64(got) - wholeBytes; d < -2 || d > 2 {
		t.Errorf("a replicated tower adds %d bytes, want the whole %.0f", got, wholeBytes)
	}
}

// Splitting the batch over the ranks lets eight images cost about one; sharding the weights
// leaves every rank encoding all eight.
func TestBatchSplitSpreadsImagesOverRanks(t *testing.T) {
	imgs := make([]EncodeImage, 8)
	for i := range imgs {
		imgs[i] = img(448, 448)
	}
	split, err := build(t, towerCase(t, 8, towerOf, &plan.EncoderRun{Resident: true, Mode: plan.EncoderBatchSplit})).
		EncodeTime(EncodeBatch{Images: imgs})
	if err != nil {
		t.Fatal(err)
	}
	shard, _ := build(t, towerCase(t, 8, towerOf, sharded)).EncodeTime(EncodeBatch{Images: imgs})
	oneSplit, _ := build(t, towerCase(t, 8, towerOf, &plan.EncoderRun{Resident: true, Mode: plan.EncoderBatchSplit})).
		EncodeTime(EncodeBatch{Images: imgs[:1]})
	if split.Total != oneSplit.Total {
		t.Errorf("eight images over eight ranks should cost one image's time: %v vs %v", split.Total, oneSplit.Total)
	}
	if shard.Total <= 0 || split.Total <= 0 {
		t.Errorf("both modes should price: %v %v", shard.Total, split.Total)
	}
}

// Where the registry carries the measured encoder-attention pair for the part and the tower's head
// dimension is one the sweep timed (64, 72, 80), the attention is priced from it and the kernel says
// so; a head dimension outside the sweep keeps the decoder's prefill pair and says that.
func TestEncoderAttentionIsMeasuredOnlyWhereTheSweepCoversIt(t *testing.T) {
	scope := func(k *Kernel) string {
		for _, o := range k.Provenance() {
			if o.Name == "encoder" {
				return o.Scope
			}
		}
		t.Fatal("no encoder assumption recorded for a resident tower")
		return ""
	}
	withHD := func(hd int) func(int) model.Tower {
		return func(h int) model.Tower {
			tw := towerOf(h)
			for i := range tw.LayerKinds[0].Nodes {
				if tw.LayerKinds[0].Nodes[i].Op == model.EncoderAttention {
					tw.LayerKinds[0].Nodes[i].HeadDim = hd
				}
			}
			return tw
		}
	}
	for _, chip := range []string{"h100", "b200"} {
		covered := build(t, towerCaseOn(t, chip, 4, withHD(72), sharded))
		if !strings.Contains(scope(covered), "measured encoder_attention pair") {
			t.Errorf("%s, head dimension 72: %q", chip, scope(covered))
		}
		outside := build(t, towerCaseOn(t, chip, 4, withHD(96), sharded))
		if !strings.Contains(scope(outside), "outside the measured encoder_attention sweep") {
			t.Errorf("%s, head dimension 96: %q", chip, scope(outside))
		}
		// Both name the part's GEMM substitution and the cap.
		for _, k := range []*Kernel{covered, outside} {
			if !strings.Contains(scope(k), "every point with an encode step is capped") {
				t.Errorf("%s: the encoder assumption does not state the cap: %q", chip, scope(k))
			}
		}
	}
}
