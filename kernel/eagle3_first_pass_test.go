package latencykernel

import "testing"

// A drafter that runs one pass over its draft positions (DSpark) is a named drafter like any
// other: its context stack carries the work over the target's tokens, and the first pass of an
// autoregressive drafter is priced over the target's scheduled tokens (priceDraft,
// draftfirstpass_test.go).
func TestAParallelBlockNamedDrafterIsNotRefusedForTheFirstPass(t *testing.T) {
	target, draft := draftGraphs()
	if _, err := withDrafter(target, draft); err != nil {
		t.Errorf("a parallel-block drafter was refused: %v", err)
	}
}
