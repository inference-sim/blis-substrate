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

// An autoregressive drafter's first pass can process one row per request or every token the
// target scheduled, and the later passes and the head can run for the requests that complete
// their prompt or for every request. Which one runs is the engine's execution decision, not a
// fact of the catalog graph, so the engine states it (Speculative.FirstPassRows and
// PassRequests) and a deployment that speculates through an autoregressive drafter without
// stating both is refused with draft_passes_unstated. The kernel names no engine and no method:
// the in-graph MTP speculators and a named EAGLE3 head are refused alike. A parallel-block
// drafter (DSpark) states its target-wide work as its context stack and needs neither field.
//
// This layer requires the statement; the layer that prices the passes reads the values. The
// tests below do not change when it does, except that a supplied value then moves the price.

func stated(s enginesettings.Speculative) *enginesettings.Speculative {
	s.FirstPassRows, s.PassRequests = enginesettings.FirstPassScheduled, enginesettings.PassRequestsAll
	return &s
}

// namedEagle3 builds the catalog pair of a target with no speculator of its own and a named
// EAGLE3 head, with the engine's speculative block as given.
func namedEagle3(t *testing.T, name string, sp *enginesettings.Speculative) (*Kernel, error) {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	draft, err := d.Model(name)
	if err != nil {
		t.Fatal(err)
	}
	c := typedRoleCase(t, "minimax-m3")
	in := c.in
	in.Model = withoutSpeculator(t, in.Model)
	in.Draft = draft
	eng := *in.Engine
	eng.Speculative = sp
	in.Engine = &eng
	return New(in)
}

func TestANamedEagle3DrafterWithUnstatedPassesIsRefused(t *testing.T) {
	for _, name := range []string{"minimax-m3-eagle3", "minimax-m3-eagle3-gqa"} {
		t.Run(name, func(t *testing.T) {
			sp := enginesettings.Speculative{Method: "eagle3", NumSpecTokens: 3, DraftModel: name}
			_, err := namedEagle3(t, name, &sp)
			if err == nil || !strings.Contains(err.Error(), "draft_passes_unstated") ||
				!strings.Contains(err.Error(), name) || !strings.Contains(err.Error(), "FirstPassRows") {
				t.Fatalf("err %v, want draft_passes_unstated naming %s and FirstPassRows", err, name)
			}
			// One field is not enough.
			sp.FirstPassRows = enginesettings.FirstPassScheduled
			if _, err := namedEagle3(t, name, &sp); err == nil || !strings.Contains(err.Error(), "PassRequests") {
				t.Errorf("err %v, want the missing PassRequests named", err)
			}
		})
	}
}

func TestANamedEagle3DrafterWithStatedPassesIsAccepted(t *testing.T) {
	sp := stated(enginesettings.Speculative{Method: "eagle3", NumSpecTokens: 3, DraftModel: "minimax-m3-eagle3"})
	if _, err := namedEagle3(t, "minimax-m3-eagle3", sp); err != nil {
		t.Fatalf("stated passes refused: %v", err)
	}
	sp.FirstPassRows, sp.PassRequests = enginesettings.FirstPassPerRequest, enginesettings.PassRequestsSampling
	if _, err := namedEagle3(t, "minimax-m3-eagle3", sp); err != nil {
		t.Fatalf("the other words refused: %v", err)
	}
}

func TestAnUnknownPassWordIsRefused(t *testing.T) {
	sp := stated(enginesettings.Speculative{Method: "eagle3", NumSpecTokens: 3, DraftModel: "minimax-m3-eagle3"})
	sp.FirstPassRows = "everything"
	if _, err := namedEagle3(t, "minimax-m3-eagle3", sp); err == nil || !strings.Contains(err.Error(), "draft_passes_unknown") {
		t.Errorf("err %v, want draft_passes_unknown", err)
	}
	sp.FirstPassRows, sp.PassRequests = enginesettings.FirstPassScheduled, "none"
	if _, err := namedEagle3(t, "minimax-m3-eagle3", sp); err == nil || !strings.Contains(err.Error(), "draft_passes_unknown") {
		t.Errorf("err %v, want draft_passes_unknown", err)
	}
}

// The model graph's own autoregressive speculator (MiniMax-M3's MTP, several tokens) is refused
// the same way, and accepted once the engine states the passes.
func TestAnInGraphAutoregressiveSpeculatorWithUnstatedPassesIsRefused(t *testing.T) {
	c := typedRoleCase(t, "minimax-m3")
	if c.in.Model.Speculator == nil {
		t.Fatal("minimax-m3 states no speculator")
	}
	run := func(sp *enginesettings.Speculative) error {
		in := c.in
		eng := *in.Engine
		eng.Speculative = sp
		in.Engine = &eng
		_, err := New(in)
		return err
	}
	for _, n := range []int{1, 3, 7} {
		err := run(&enginesettings.Speculative{Method: "mtp", NumSpecTokens: n})
		if err == nil || !strings.Contains(err.Error(), "draft_passes_unstated") {
			t.Errorf("K=%d: err %v, want draft_passes_unstated", n, err)
		}
		if err := run(stated(enginesettings.Speculative{Method: "mtp", NumSpecTokens: n})); err != nil {
			t.Errorf("K=%d: stated passes refused: %v", n, err)
		}
	}
	if err := run(nil); err != nil {
		t.Errorf("no speculation was refused: %v", err)
	}
}

// The refusal names neither an engine nor a method.
func TestTheDraftPassesRefusalNamesNoEngineOrMethod(t *testing.T) {
	c := typedRoleCase(t, "minimax-m3")
	in := c.in
	eng := *in.Engine
	eng.Speculative = &enginesettings.Speculative{Method: "mtp", NumSpecTokens: 3}
	in.Engine = &eng
	_, err := New(in)
	if err == nil {
		t.Fatal("not refused")
	}
	for _, w := range []string{"vllm", "vLLM", "sglang", "mtp", "eagle"} {
		if strings.Contains(err.Error(), w) {
			t.Errorf("refusal %q names %q", err, w)
		}
	}
}

// A parallel-block drafter needs neither field, and a deployment that does not speculate is
// unaffected whatever the fields say.
func TestAParallelBlockDrafterNeedsNoPassStatement(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	draft, err := d.Model("kimi-k3-dspark")
	if err != nil {
		t.Fatal(err)
	}
	c := typedRoleCase(t, "kimi-k3")
	in := c.in
	in.Draft = draft
	eng := *in.Engine
	eng.Speculative = &enginesettings.Speculative{Method: "dspark", NumSpecTokens: 7, DraftModel: "kimi-k3-dspark"}
	in.Engine = &eng
	if _, err := New(in); err != nil {
		t.Errorf("a parallel-block drafter with no pass statement was refused: %v", err)
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
