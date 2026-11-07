package coefficient

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

var (
	implA = KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "aaaaaaaaaaaa"}
	implB = KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "bbbbbbbbbbbb"}
	// implOtherVariant is the same build of the same library as implA, another kernel of it.
	implOtherVariant = KernelImpl{Library: "nccl", Variant: "ncclAllGather", CodeHash: "aaaaaaaaaaaa"}
	implFlash        = KernelImpl{Library: "flashinfer", Variant: "decode", CodeHash: "cccccccccccc"}
)

func keyed(impls ...KernelImpl) Entry {
	return Entry{Name: "x", Method: vocab.MethodMeasured, KernelImpls: impls}
}

// VAL-8: a keyed entry matches only the implementation that runs, by library, variant and code
// hash together, and there is no fallback to a sibling.
func TestMatchKernels(t *testing.T) {
	for _, c := range []struct {
		name    string
		e       Entry
		running []KernelImpl
		want    KernelMatch
	}{
		{"the listed implementation runs", keyed(implA), []KernelImpl{implA}, KernelExact},
		{"it runs among others", keyed(implA), []KernelImpl{implFlash, implA}, KernelExact},
		{"one of several listed runs", keyed(implA, implB), []KernelImpl{implB}, KernelExact},
		{"a different hash of the same kernel", keyed(implA), []KernelImpl{implB}, KernelRefused},
		{"the same hash of another variant", keyed(implA), []KernelImpl{implOtherVariant}, KernelRefused},
		{"another library", keyed(implA), []KernelImpl{implFlash}, KernelRefused},
		{"nothing is said to run", keyed(implA), nil, KernelRefused},
		{"the engine could not identify the build", keyed(implA),
			[]KernelImpl{{Library: "nccl", Variant: "ncclAllReduce"}}, KernelRefused},
		{"a toolkit is not part of the key", keyed(implA), []KernelImpl{implA}, KernelExact},
		{"an entry that lists none, nothing runs", keyed(), nil, KernelUnkeyed},
		{"an entry that lists none applies to whatever runs", keyed(), []KernelImpl{implB}, KernelUnkeyed},
	} {
		if got := c.e.MatchKernels(c.running); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

// A transferred value is valid for its target and measured on its source. The lookup reads the
// target only: a transferred entry matches the implementation it is moved to and not the one it
// was measured on.
func TestATransferredEntryMatchesItsTargetAndNotItsSource(t *testing.T) {
	e := Entry{Name: "x", Method: vocab.MethodTransferred, KernelImpls: []KernelImpl{implB}, MeasuredOn: []KernelImpl{implA}}
	if got := e.MatchKernels([]KernelImpl{implB}); got != KernelExact {
		t.Errorf("the target: got %q", got)
	}
	if got := e.MatchKernels([]KernelImpl{implA}); got != KernelRefused {
		t.Errorf("the source implementation: got %q, want refused (measured_on is never matched)", got)
	}
	if got := e.SourceImpls(); len(got) != 1 || got[0] != implA {
		t.Errorf("source: %v", got)
	}
	if got := e.TargetImpls(); len(got) != 1 || got[0] != implB {
		t.Errorf("target: %v", got)
	}
	m := keyed(implA)
	if got := m.SourceImpls(); len(got) != 1 || got[0] != implA {
		t.Errorf("a measured entry's source is its key (D5): %v", got)
	}
	a := Entry{Method: vocab.MethodAssumed}
	if a.SourceImpls() != nil {
		t.Errorf("an assumed entry has no source implementation")
	}
}

func TestImplsCompatible(t *testing.T) {
	for _, c := range []struct {
		name string
		a, b []KernelImpl
		want bool
	}{
		{"an unkeyed entry beside a keyed one", nil, []KernelImpl{implA}, true},
		{"two unkeyed entries", nil, nil, true},
		{"the same implementation", []KernelImpl{implA}, []KernelImpl{implA}, true},
		{"two builds of one kernel never both run", []KernelImpl{implA}, []KernelImpl{implB}, false},
		{"two kernels of one library can both run", []KernelImpl{implA}, []KernelImpl{implOtherVariant}, true},
		{"different libraries can both run", []KernelImpl{implA}, []KernelImpl{implFlash}, true},
		{"a list with one compatible pair", []KernelImpl{implA, implFlash}, []KernelImpl{implB, implFlash}, true},
		{"a list whose every pair clashes", []KernelImpl{implA}, []KernelImpl{implB, implB}, false},
	} {
		if got := ImplsCompatible(keyed(c.a...), keyed(c.b...)); got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
		if got := ImplsCompatible(keyed(c.b...), keyed(c.a...)); got != c.want {
			t.Errorf("%s, reversed: got %v, want %v", c.name, got, c.want)
		}
	}
}

// The rule by method is one function, and the table of it is the documentation.
func TestMethodsThatMayKeyAnEntry(t *testing.T) {
	want := map[vocab.Method][2]bool{ // {may state kernel_impls, may state measured_on}
		vocab.MethodMeasured: {true, false}, vocab.MethodCalibrated: {true, false},
		vocab.MethodTransferred: {true, true}, vocab.MethodLiterature: {true, false},
		vocab.MethodVendorSpec: {false, false}, vocab.MethodRoofline: {false, false},
		vocab.MethodAssumed: {false, false}, vocab.MethodNotCharged: {false, false},
	}
	for _, ms := range vocab.AllMethods() {
		m := vocab.Method(ms)
		w, ok := want[m]
		if !ok {
			t.Errorf("method %q is not in the table", m)
			continue
		}
		if MayStateKernelImpls(m) != w[0] || MayStateMeasuredOn(m) != w[1] {
			t.Errorf("%s: kernel_impls %v measured_on %v, want %v", m, MayStateKernelImpls(m), MayStateMeasuredOn(m), w)
		}
	}
}

// Two entries of one name, scope and method that differ only in the build they are valid for
// are two coefficients: a kernel whose source changed gets its own value.
func TestEntriesForTwoBuildsOfOneKernelAreBothValidInOneSet(t *testing.T) {
	s := validSet()
	a, b := s.Coefficients[0], s.Coefficients[0]
	a.KernelImpls, b.KernelImpls = []KernelImpl{implA}, []KernelImpl{implB}
	b.Value = 0.5
	s.Coefficients = []Entry{a, b}
	if p := s.Validate(); !p.OK() {
		t.Fatalf("two builds of one kernel were rejected:\n%s", p.Error())
	}
	b.KernelImpls = []KernelImpl{implA}
	s.Coefficients = []Entry{a, b}
	if p := s.Validate(); p.OK() || !strings.Contains(p.Error(), "duplicate entry name") {
		t.Fatalf("two entries with one key must be a duplicate, got %v", p)
	}
	// Listing the implementations in another order is the same identity.
	a.KernelImpls = []KernelImpl{implA, implB}
	b.KernelImpls = []KernelImpl{implB, implA}
	s.Coefficients = []Entry{a, b}
	if p := s.Validate(); p.OK() {
		t.Fatalf("the order of kernel_impls must not make two entries different")
	}
}

func TestValidateRunning(t *testing.T) {
	for _, c := range []struct {
		name    string
		impls   []KernelImpl
		wantErr string
	}{
		{"none", nil, ""},
		{"one build of each kernel", []KernelImpl{implA, implFlash}, ""},
		{"a build the engine could not identify", []KernelImpl{{Library: "nccl", Variant: "ncclAllReduce"}}, ""},
		{"a library alone", []KernelImpl{{Library: "cublas"}}, ""},
		{"the same implementation twice", []KernelImpl{implA, implA}, ""},
		{"no library", []KernelImpl{{Variant: "v", CodeHash: "aaaaaaaaaaaa"}}, "library is required"},
		{"a hash with no variant", []KernelImpl{{Library: "l", CodeHash: "aaaaaaaaaaaa"}}, "variant"},
		{"a malformed hash", []KernelImpl{{Library: "l", Variant: "v", CodeHash: "AAAA"}}, "lower-case hexadecimal"},
		{"two builds of one kernel", []KernelImpl{implA, implB}, "two builds of one kernel"},
	} {
		err := ValidateRunning(c.impls)
		switch {
		case c.wantErr == "" && err != nil:
			t.Errorf("%s: %v", c.name, err)
		case c.wantErr != "" && (err == nil || !strings.Contains(err.Error(), c.wantErr)):
			t.Errorf("%s: got %v, want an error containing %q", c.name, err, c.wantErr)
		}
	}
}
