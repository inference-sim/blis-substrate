package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
	"github.com/inference-sim/blis-substrate/load"
)

// VAL-8 at the pricing boundary: a coefficient that lists kernel implementations prices a step
// only when one of them is what the pool runs (Inputs.Implementations, the PoolPlan's
// Implementations()). The test builds a real kernel, keys the entry that moves its price to one
// build, and adds a transferred value for another.

var (
	buildA = plan.KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "aaaaaaaaaaaa", Toolkit: "cuda13.0"}
	buildB = plan.KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "bbbbbbbbbbbb", Toolkit: "cuda13.0"}
	buildC = plan.KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "cccccccccccc"}
)

// keyedPair returns the sets with the entry keyed to build A (measured), and a transferred value
// for build B, measured on A, at 0.8 times the measured value.
func keyedPair(sets []*coefficient.Set, r resolve.Resolved) []*coefficient.Set {
	implA := coefficient.KernelImpl{Library: buildA.Library, Variant: buildA.Variant, CodeHash: buildA.CodeHash}
	implB := coefficient.KernelImpl{Library: buildB.Library, Variant: buildB.Variant, CodeHash: buildB.CodeHash}
	out := make([]*coefficient.Set, len(sets))
	for i, s := range sets {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		if s.Name == r.Set {
			for j, e := range c.Coefficients {
				if e.Identity() != r.Entry.Identity() {
					continue
				}
				keyed := e
				keyed.KernelImpls = []coefficient.KernelImpl{implA}
				c.Coefficients[j] = keyed
				moved := e
				moved.Value = e.Value * 0.8
				moved.Method, moved.Fitted = vocab.MethodTransferred, false
				moved.KernelImpls, moved.MeasuredOn = []coefficient.KernelImpl{implB}, []coefficient.KernelImpl{implA}
				moved.TransferredFrom, moved.TransferRule = "hardware=h200", "a new build of the same kernel, same shapes"
				moved.Rationale = "test"
				moved.Measures, moved.FitRange, moved.CI95, moved.Sources = nil, nil, nil, nil
				c.Coefficients = append(c.Coefficients, moved)
				break
			}
		}
		out[i] = &c
	}
	return out
}

func TestTheKernelImplementationThatRunsDecidesWhichCoefficientPrices(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a catalog scenario for several kernels")
	}
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	cases := perturbationScenarios(t, d, d.Models()[0])
	if len(cases) == 0 {
		t.Fatal("no scenario builds")
	}
	in := cases[0].in
	base, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	want := pricesOf(base)
	// The entry to key: a measured one, unscoped to a fabric, whose value moves a price.
	var pick resolve.Resolved
	var movedBy string
	for _, sel := range base.selected {
		if sel.Entry.Method != vocab.MethodMeasured || len(sel.Entry.Scope.Fabric) > 0 || sel.Entry.Value == 0 {
			continue
		}
		scaled := in
		scaled.Coefficients = perturbed(in.Coefficients, sel.Entry.Name)
		if k, err := New(scaled); err == nil && pricesOf(k) != want {
			pick, movedBy = sel, pricesOf(k)
			break
		}
	}
	if pick.Entry.Name == "" {
		t.Fatal("no measured entry moves the price of the scenario")
	}
	name := pick.Entry.Name
	keyed := in
	keyed.Coefficients = keyedPair(in.Coefficients, pick)
	build := func(impls ...plan.KernelImpl) (*Kernel, error) {
		x := keyed
		x.Implementations = impls
		return New(x)
	}
	matchOf := func(k *Kernel) string {
		for _, o := range k.Provenance() {
			if o.Name == name && o.Set != KernelAssumptionSet {
				return o.KernelMatch
			}
		}
		return "missing"
	}

	// The build the measured entry is keyed to: it prices, as it did before it was keyed.
	k, err := build(buildA)
	if err != nil {
		t.Fatalf("build A: %v", err)
	}
	if pricesOf(k) != want || matchOf(k) != "exact" {
		t.Errorf("build A: prices equal the unkeyed kernel's = %v, match %q", pricesOf(k) == want, matchOf(k))
	}

	// Transferred evidence for a known target: the target prices from the transferred value,
	// 0.8 times the measured one, and only it.
	k, err = build(buildB)
	if err != nil {
		t.Fatalf("build B: %v", err)
	}
	if pricesOf(k) != movedBy || matchOf(k) != "exact" {
		t.Errorf("build B: prices come from the transferred value = %v, match %q", pricesOf(k) == movedBy, matchOf(k))
	}

	// A different build, and no build stated: neither entry matches. There is no sibling
	// fallback: the kernel either refuses, naming what it refused, or, for a coefficient it reads
	// as optional, prices without it. It never prices from the value of another build.
	for label, impls := range map[string][]plan.KernelImpl{"build C": {buildC}, "none stated": nil,
		"another kernel": {{Library: "flashinfer", Variant: "decode", CodeHash: "dddddddddddd"}}} {
		k, err := build(impls...)
		switch {
		case err != nil:
			if !strings.Contains(err.Error(), "VAL-8") || !strings.Contains(err.Error(), "aaaaaaaaaaaa") {
				t.Errorf("%s: the refusal should name VAL-8 and the build refused: %v", label, err)
			}
		case pricesOf(k) == want || pricesOf(k) == movedBy:
			t.Errorf("%s: priced from a build it does not run", label)
		case matchOf(k) != "missing":
			t.Errorf("%s: %s is still in the provenance with match %q", label, name, matchOf(k))
		}
	}

	// The kernel is built for one pool: two builds of one kernel cannot both run.
	if _, err := build(buildA, buildB); err == nil || !strings.Contains(err.Error(), "two builds of one kernel") {
		t.Errorf("two builds of one kernel: %v", err)
	}
	// A malformed hash is refused rather than quietly matching nothing.
	if _, err := build(plan.KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "AAAA"}); err == nil {
		t.Errorf("a malformed hash was accepted")
	}

	// The registry as committed lists no kernel implementation: stating what runs changes no
	// price, and every coefficient is reported unkeyed.
	stated := in
	stated.Implementations = []plan.KernelImpl{buildA, {Library: "cublas", Variant: "gemm", CodeHash: "eeeeeeeeeeee"}}
	k, err = New(stated)
	if err != nil {
		t.Fatal(err)
	}
	if pricesOf(k) != want {
		t.Errorf("stating the running implementations changed a price while no entry lists one")
	}
	for _, o := range k.Provenance() {
		if o.Set != KernelAssumptionSet && o.KernelMatch != "unkeyed" {
			t.Errorf("%s: match %q, want unkeyed", o.Name, o.KernelMatch)
		}
		if o.Set == KernelAssumptionSet && o.KernelMatch != "" {
			t.Errorf("%s: the kernel's own assumption has match %q", o.Name, o.KernelMatch)
		}
	}
}
