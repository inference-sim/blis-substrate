package latencykernel

import (
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
	"gopkg.in/yaml.v3"
)

// The first pass of an EAGLE-style drafter runs its layers over every token the target
// scheduled that step, not one row per request (vllm/v1/worker/gpu/spec_decode/autoregressive/
// speculator.py _prefill, :453-478 at v0.31.0; blis-schemas#60). The cost model prices each
// autoregressive pass at one row per request, so a price for such a drafter would leave that
// pass out with nothing in the result saying so. A drafter named by the deployment (EAGLE3
// heads are checkpoints of their own) is therefore refused, with the issue's number in the
// reason, until the pass is priced.
//
// LIFTING THIS REFUSAL (blis-substrate layer 11, which prices the pass): delete
// refuseUnpricedFirstDraftPass and its call in withDrafter, delete this file's two refusal
// tests, and add a test that the first pass of a named EAGLE3 drafter is charged over the
// target's scheduled tokens (fails if it is priced at one row per request).

// A catalog drafter that is a separate autoregressive checkpoint, joined to a target that
// states no speculator of its own. Before the refusal this priced.
func TestANamedEagle3DrafterIsRefusedWhileItsFirstPassIsUnpriced(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"minimax-m3-eagle3", "minimax-m3-eagle3-gqa"} {
		t.Run(name, func(t *testing.T) {
			draft, err := d.Model(name)
			if err != nil {
				t.Fatal(err)
			}
			c := typedRoleCase(t, "minimax-m3")
			in := c.in
			in.Model = withoutSpeculator(t, in.Model)
			in.Draft = draft
			eng := *in.Engine
			eng.Speculative = &enginesettings.Speculative{Method: "eagle3", NumSpecTokens: 3, DraftModel: name}
			in.Engine = &eng
			k, err := New(in)
			if err == nil {
				t.Fatalf("%s priced with no refusal and no marker: its first pass over the target's "+
					"scheduled tokens is not in the price (kernel %v)", name, k != nil)
			}
			for _, want := range []string{"blis-schemas#60", "first pass", name} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("refusal %q does not name %q", err, want)
				}
			}
		})
	}
}

// The refusal is about the pass, so it applies before the other checks of the graft: a target
// that states a speculator is refused for the pass, which is the reason that outlives the
// replacement layer 11 adds.
func TestTheFirstPassRefusalComesBeforeTheGraftChecks(t *testing.T) {
	target, draft := draftGraphs()
	draft.Speculator.Method = "eagle3"
	draft.Speculator.Drafting = model.DraftingAutoregressive
	target.Speculator = draft.Speculator // would be refused as "already states" otherwise
	if _, err := withDrafter(target, draft); err == nil || !strings.Contains(err.Error(), "blis-schemas#60") {
		t.Errorf("err %v, want the first-pass refusal", err)
	}
}

// A drafter that runs one pass over its draft positions (DSpark) is not refused: its context
// stack carries the work over the target's tokens.
func TestAParallelBlockNamedDrafterIsNotRefusedForTheFirstPass(t *testing.T) {
	target, draft := draftGraphs()
	if _, err := withDrafter(target, draft); err != nil {
		t.Errorf("a parallel-block drafter was refused: %v", err)
	}
}

// withoutSpeculator returns a copy of g with its speculator and the layer kinds only the
// speculator ran removed.
func withoutSpeculator(t *testing.T, g *model.Graph) *model.Graph {
	t.Helper()
	b, err := yaml.Marshal(*g)
	if err != nil {
		t.Fatal(err)
	}
	var c model.Graph
	if err := yaml.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	used := map[string]bool{}
	for _, id := range c.Stack.Expand() {
		used[id] = true
	}
	c.Speculator = nil
	if len(c.Embedding) == 0 {
		// An EAGLE3 head aliases the target's token embedding; the MiniMax-M3 graphs state none
		// in this stack, so the copy is given the one vLLM builds, to let the pair validate.
		var gather model.Node
		node := "op: Gather\nrole: embed_tokens\nrows: " + strconv.Itoa(c.Global.VocabSize) +
			"\nrow_bytes: " + strconv.Itoa(2*c.Global.HiddenSize) +
			"\nrows_per_token: 1\nout_bytes_per_token: " + strconv.Itoa(2*c.Global.HiddenSize) +
			"\nshard: rows\n"
		if err := yaml.Unmarshal([]byte(node), &gather); err != nil {
			t.Fatal(err)
		}
		c.Embedding = []model.Node{gather}
	}
	kinds := c.LayerKinds[:0]
	for _, lk := range c.LayerKinds {
		if used[lk.ID] {
			kinds = append(kinds, lk)
		}
	}
	c.LayerKinds = kinds
	return &c
}
