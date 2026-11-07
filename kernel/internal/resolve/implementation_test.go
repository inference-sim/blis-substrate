package resolve

import (
	"errors"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/validate/dataset"
)

// VAL-8: a lookup matches a coefficient only for the exact kernel implementation that runs.

var (
	nccl = func(hash string) coefficient.KernelImpl {
		return coefficient.KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: hash}
	}
	hashA, hashB = nccl("aaaaaaaaaaaa"), nccl("bbbbbbbbbbbb")
)

// h100 is the scope the review's regression used.
var h100 = coefficient.Scope{Hardware: []string{"h100"}}

func measuredKeyed(v float64, impls ...coefficient.KernelImpl) coefficient.Entry {
	return coefficient.Entry{Name: "x", Value: v, Units: vocab.UnitDimensionless, Method: vocab.MethodMeasured,
		Fitted: true, Scope: h100, KernelImpls: impls}
}

// transferredKeyed is a value measured on source and moved to target by a stated rule.
func transferredKeyed(v float64, target, source coefficient.KernelImpl) coefficient.Entry {
	return coefficient.Entry{Name: "x", Value: v, Units: vocab.UnitDimensionless, Method: vocab.MethodTransferred,
		Scope: h100, KernelImpls: []coefficient.KernelImpl{target}, MeasuredOn: []coefficient.KernelImpl{source},
		TransferredFrom: "hardware=h100", TransferRule: "same kernel family, new build", Rationale: "r"}
}

func assumedFallback(v float64) coefficient.Entry {
	return coefficient.Entry{Name: "x", Value: v, Units: vocab.UnitDimensionless, Method: vocab.MethodAssumed,
		Scope: h100, Rationale: "r"}
}

func setOf(es ...coefficient.Entry) []*coefficient.Set {
	return []*coefficient.Set{{Kind: "CoefficientSet", Name: "s", Coefficients: es}}
}

func on(impls ...coefficient.KernelImpl) Scope {
	return Scope{Hardware: "h100", Implementations: impls}
}

// valueAt looks "x" up for a deployment and returns the entry's value and how it matched.
func valueAt(t *testing.T, sets []*coefficient.Set, d Scope) (float64, coefficient.KernelMatch, error) {
	t.Helper()
	c, err := Load(sets, d)
	if err != nil {
		return 0, "", err
	}
	v, err := c.Value("x")
	if err != nil {
		return 0, "", err
	}
	r, _ := c.Entry("x")
	return v, r.Match, nil
}

// The review's regression: an H100 measured entry keyed to aaaaaaaaaaaa was returned for a
// target at bbbbbbbbbbbb.
func TestAMeasuredEntryIsNotReturnedForAnotherBuildOfItsKernel(t *testing.T) {
	sets := setOf(measuredKeyed(1, hashA))
	if v, m, err := valueAt(t, sets, on(hashA)); err != nil || v != 1 || m != coefficient.KernelExact {
		t.Fatalf("the build the entry is keyed to: value %v match %q err %v", v, m, err)
	}
	_, _, err := valueAt(t, sets, on(hashB))
	if err == nil {
		t.Fatalf("the entry keyed to aaaaaaaaaaaa was returned for bbbbbbbbbbbb")
	}
	for _, want := range []string{"VAL-8", "aaaaaaaaaaaa", "bbbbbbbbbbbb"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the error should name %q:\n%v", want, err)
		}
	}
}

// No fallback to a sibling: the same hash of another variant, another library, and the same
// kernel with no hash are all refused.
func TestNoSiblingImplementationMatches(t *testing.T) {
	sets := setOf(measuredKeyed(1, hashA))
	for name, running := range map[string][]coefficient.KernelImpl{
		"another variant":                     {{Library: "nccl", Variant: "ncclAllGather", CodeHash: "aaaaaaaaaaaa"}},
		"another library":                     {{Library: "rccl", Variant: "ncclAllReduce", CodeHash: "aaaaaaaaaaaa"}},
		"the kernel with no identified build": {{Library: "nccl", Variant: "ncclAllReduce"}},
		"nothing stated":                      nil,
	} {
		if _, _, err := valueAt(t, sets, on(running...)); err == nil {
			t.Errorf("%s: the entry keyed to aaaaaaaaaaaa was returned", name)
		}
	}
}

// When another entry of the lookup survives, a refused name is missing from the resolved set and
// the error says which entries were refused and why.
func TestAMissingCoefficientNamesTheEntriesRefusedForItsImplementation(t *testing.T) {
	other := measuredKeyed(3)
	other.Name = "y"
	c, err := Load(setOf(measuredKeyed(1, hashA), other), on(hashB))
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Value("x")
	var missing *MissingError
	if !errors.As(err, &missing) || !strings.Contains(missing.Refused, "aaaaaaaaaaaa") ||
		!strings.Contains(err.Error(), "bbbbbbbbbbbb") {
		t.Fatalf("got %v", err)
	}
}

// A refused entry lets the next entry of the lookup win: the design's fallback, at worst the
// family's assumed entry, and the price records that method (VAL-3).
func TestARefusedEntryFallsToTheNextRankNotToASibling(t *testing.T) {
	sets := setOf(measuredKeyed(1, hashA), assumedFallback(7))
	v, m, err := valueAt(t, sets, on(hashB))
	if err != nil || v != 7 || m != coefficient.KernelUnkeyed {
		t.Fatalf("got value %v match %q err %v, want the assumed 7 unkeyed", v, m, err)
	}
	v, m, err = valueAt(t, sets, on(hashA))
	if err != nil || v != 1 || m != coefficient.KernelExact {
		t.Fatalf("got value %v match %q err %v, want the measured 1 exact", v, m, err)
	}
	c, _ := Load(sets, on(hashB))
	got := c.Refused()
	if len(got) != 1 || got[0].Entry.Value != 1 || len(got[0].Running) != 1 || got[0].Running[0] != hashB {
		t.Fatalf("the refusal is recorded with the build that ran in its slot: %+v", got)
	}
}

// Transferred evidence for a known target: measured on one build, valid for another, matching
// only that other. The measured_on source is never matched.
func TestATransferredEntryMatchesOnlyItsTarget(t *testing.T) {
	hashC := nccl("cccccccccccc")
	sets := setOf(transferredKeyed(2, hashB, hashA), assumedFallback(7))
	if v, m, err := valueAt(t, sets, on(hashB)); err != nil || v != 2 || m != coefficient.KernelExact {
		t.Fatalf("the target: value %v match %q err %v", v, m, err)
	}
	for name, d := range map[string]Scope{"the source build": on(hashA), "a third build": on(hashC), "unset": on()} {
		if v, _, err := valueAt(t, sets, d); err != nil || v != 7 {
			t.Errorf("%s: value %v err %v, want the assumed 7: the transferred value is valid for its target only", name, v, err)
		}
	}
}

// Two builds of a kernel are two coefficients: each matches its own and the other is refused,
// in both load orders.
func TestEachBuildOfAKernelGetsItsOwnValue(t *testing.T) {
	ea, eb := measuredKeyed(1, hashA), measuredKeyed(2, hashB)
	for _, es := range [][]coefficient.Entry{{ea, eb}, {eb, ea}} {
		sets := setOf(es...)
		for hash, want := range map[coefficient.KernelImpl]float64{hashA: 1, hashB: 2} {
			if v, _, err := valueAt(t, sets, on(hash)); err != nil || v != want {
				t.Errorf("build %s: value %v err %v, want %v", hash.CodeHash, v, err, want)
			}
		}
	}
}

// Unset: only an entry that states no implementation matches, and it is reported unkeyed, so a
// reader never counts it as shown to match the kernel that runs. Every price that did not
// depend on an implementation is unchanged.
func TestWithNoImplementationStatedOnlyUnkeyedEntriesMatch(t *testing.T) {
	sets := setOf(measuredKeyed(1, hashA), func() coefficient.Entry {
		e := measuredKeyed(3)
		e.Name = "y"
		return e
	}())
	c, err := Load(sets, on())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Value("x"); err == nil {
		t.Errorf("the keyed entry was returned with no implementation stated")
	}
	if v, err := c.Value("y"); err != nil || v != 3 {
		t.Errorf("an unkeyed entry: %v %v", v, err)
	}
	if r, _ := c.Entry("y"); r.Match != coefficient.KernelUnkeyed {
		t.Errorf("match = %q, want unkeyed", r.Match)
	}
	// With an implementation stated, the unkeyed entry still applies to it and is still unkeyed.
	c, _ = Load(sets, on(hashB))
	if r, ok := c.Entry("y"); !ok || r.Match != coefficient.KernelUnkeyed {
		t.Errorf("an unkeyed entry under a stated implementation: %+v %v", r, ok)
	}
}

// X15: the registry validator and the lookup agree about which keyed entries can be chosen
// between. For every pair of entries of one name, scope and method that differ only in their
// kernel key, and every running set a pool can have, the lookup refuses exactly when the
// validator flags the pair, in both load orders.
func TestLookupAgreesWithTheValidatorOnKernelKeys(t *testing.T) {
	other := coefficient.KernelImpl{Library: "nccl", Variant: "ncclAllGather", CodeHash: "aaaaaaaaaaaa"}
	flash := coefficient.KernelImpl{Library: "flashinfer", Variant: "decode", CodeHash: "cccccccccccc"}
	keys := [][]coefficient.KernelImpl{nil, {hashA}, {hashB}, {other}, {flash}, {hashA, flash}, {hashB, flash}}
	// Running sets a pool can have: at most one build of a kernel.
	var runnings [][]coefficient.KernelImpl
	runnings = append(runnings, nil)
	for _, a := range []*coefficient.KernelImpl{nil, &hashA, &hashB} {
		for _, o := range []*coefficient.KernelImpl{nil, &other} {
			for _, f := range []*coefficient.KernelImpl{nil, &flash} {
				var r []coefficient.KernelImpl
				for _, p := range []*coefficient.KernelImpl{a, o, f} {
					if p != nil {
						r = append(r, *p)
					}
				}
				runnings = append(runnings, r)
			}
		}
	}
	var accepted, refused, apart int
	for i, ka := range keys {
		for _, kb := range keys[i:] {
			// A pair of one rank is unordered when both match; a pair of two ranks is ordered by
			// the rank, so some pairs the validator accepts are matched together too.
			for _, bMethod := range []vocab.Method{vocab.MethodMeasured, vocab.MethodTransferred} {
				a, b := measuredKeyed(1, ka...), measuredKeyed(2, kb...)
				b.Method = bMethod
				sa := &coefficient.Set{Kind: "CoefficientSet", Name: "a", Coefficients: []coefficient.Entry{a}}
				sb := &coefficient.Set{Kind: "CoefficientSet", Name: "b", Coefficients: []coefficient.Entry{b}}
				flagged := len(dataset.CheckOrdered(map[string]*coefficient.Set{"a": sa, "b": sb})) > 0
				witnessed := false
				for _, run := range runnings {
					both := a.MatchKernels(run) != coefficient.KernelRefused && b.MatchKernels(run) != coefficient.KernelRefused
					if !both {
						apart++
						continue
					}
					witnessed = true
					for _, order := range [][]*coefficient.Set{{sa, sb}, {sb, sa}} {
						_, err := Load(order, Scope{Hardware: "h100", Implementations: run})
						if (err != nil) != flagged {
							t.Fatalf("keys %v and %v running %v: the validator flags the pair = %v, the lookup error = %v",
								ka, kb, run, flagged, err)
						}
					}
					if flagged {
						refused++
					} else {
						accepted++
					}
				}
				// The other direction: a pair the validator flags must be one some pool can match
				// together, or the registry refuses data that no lookup would ever find ambiguous.
				if flagged && !witnessed {
					t.Fatalf("keys %v and %v: the validator flags a pair that no running set matches together", ka, kb)
				}
			}
		}
	}
	if accepted == 0 || refused == 0 || apart == 0 {
		t.Fatalf("not exercised: %d accepted, %d refused, %d not matching one running set", accepted, refused, apart)
	}
}
