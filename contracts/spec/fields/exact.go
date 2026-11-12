package fields

import (
	"regexp"
	"strconv"
	"strings"
)

// Clause is one way a stand-in can be exact (design §11.2). It holds when every key it sets
// holds, so a clause is a conjunction. At least one key is set.
type Clause struct {
	// ValueIn: the recorded value equals one of the strings, ignoring case.
	ValueIn []string `json:"value_in,omitempty" yaml:"value_in,omitempty"`
	// ValueListHasAny: the recorded value, split on commas, has at least one of the strings
	// as an item.
	ValueListHasAny []string `json:"value_list_has_any,omitempty" yaml:"value_list_has_any,omitempty"`
	// ValueListHasNone: the recorded value, split on commas, has none of the strings as an item.
	ValueListHasNone []string `json:"value_list_has_none,omitempty" yaml:"value_list_has_none,omitempty"`
	// ValueNotPrefixed: the recorded value starts with none of the strings.
	ValueNotPrefixed []string `json:"value_not_prefixed,omitempty" yaml:"value_not_prefixed,omitempty"`
	// TPGroupNodes: the number of nodes the pool's tensor-parallel group spans satisfies
	// every comparison.
	TPGroupNodes *NodeComparison `json:"tp_group_nodes,omitempty" yaml:"tp_group_nodes,omitempty"`
	// ComputeCapability: the GPU part's compute capability satisfies every comparison.
	ComputeCapability *CapabilityComparison `json:"compute_capability,omitempty" yaml:"compute_capability,omitempty"`
}

// NodeComparison compares a node count. It holds when every comparison it sets holds; at
// least one is set.
type NodeComparison struct {
	Eq  *int `json:"eq,omitempty" yaml:"eq,omitempty"`
	Gt  *int `json:"gt,omitempty" yaml:"gt,omitempty"`
	Gte *int `json:"gte,omitempty" yaml:"gte,omitempty"`
	Lt  *int `json:"lt,omitempty" yaml:"lt,omitempty"`
	Lte *int `json:"lte,omitempty" yaml:"lte,omitempty"`
}

// CapabilityComparison compares a compute capability, each side a "major.minor" string such
// as "10.0". It holds when every comparison it sets holds; at least one is set.
type CapabilityComparison struct {
	Eq  *string `json:"eq,omitempty" yaml:"eq,omitempty"`
	Gt  *string `json:"gt,omitempty" yaml:"gt,omitempty"`
	Gte *string `json:"gte,omitempty" yaml:"gte,omitempty"`
	Lt  *string `json:"lt,omitempty" yaml:"lt,omitempty"`
	Lte *string `json:"lte,omitempty" yaml:"lte,omitempty"`
}

// Facts are what a consumer knows about a recorded value when it asks Exact. A fact the
// consumer cannot supply is left at its zero value, and a clause that needs it does not hold.
type Facts struct {
	// Value is the recorded value. Exact trims it and folds it to lower case.
	Value string
	// TPGroupNodes is the number of nodes the pool's tensor-parallel group spans, from the
	// pool's tensor-parallel size and the GPUs per node. Nil when unknown.
	TPGroupNodes *int
	// ComputeCapability is the GPU part's compute capability, "major.minor", from the
	// catalog. Empty when unknown.
	ComputeCapability string
}

// Exact reports whether the estimation rule's stand-in is exact for the recorded value: the
// upstream resolves the field, left unset or set to the stand-in, as it resolves the value.
//
// A rule that is not leave_unset or substitute has no stand-in, so it is never exact. A rule
// with no valid_when applies to every value. Otherwise the stand-in is exact when at least
// one clause holds. A value for which none does must not be stood in for.
func Exact(e Estimation, f Facts) bool {
	if e.Rule != RuleLeaveUnset && e.Rule != RuleSubstitute {
		return false
	}
	if e.ValidWhen == nil {
		return true
	}
	v := strings.ToLower(strings.TrimSpace(f.Value))
	for _, c := range e.ValidWhen {
		if c.holds(v, f) {
			return true
		}
	}
	return false
}

func (c Clause) holds(v string, f Facts) bool {
	var items map[string]bool
	if c.ValueListHasAny != nil || c.ValueListHasNone != nil {
		items = map[string]bool{}
		for _, it := range strings.Split(v, ",") {
			if it = strings.TrimSpace(it); it != "" {
				items[it] = true
			}
		}
	}
	if c.ValueIn != nil && !oneOf(v, c.ValueIn) {
		return false
	}
	if c.ValueListHasAny != nil {
		found := false
		for _, s := range c.ValueListHasAny {
			found = found || items[fold(s)]
		}
		if !found {
			return false
		}
	}
	if c.ValueListHasNone != nil {
		for _, s := range c.ValueListHasNone {
			if items[fold(s)] {
				return false
			}
		}
	}
	for _, p := range c.ValueNotPrefixed {
		if strings.HasPrefix(v, fold(p)) {
			return false
		}
	}
	if c.TPGroupNodes != nil && (f.TPGroupNodes == nil || !c.TPGroupNodes.holds(*f.TPGroupNodes)) {
		return false
	}
	if c.ComputeCapability != nil {
		have, ok := parseCapability(f.ComputeCapability)
		if !ok || !c.ComputeCapability.holds(have) {
			return false
		}
	}
	return true
}

func fold(s string) string { return strings.ToLower(strings.TrimSpace(s)) }

func oneOf(v string, set []string) bool {
	for _, s := range set {
		if v == fold(s) {
			return true
		}
	}
	return false
}

func (n NodeComparison) holds(have int) bool {
	return (n.Eq == nil || have == *n.Eq) && (n.Gt == nil || have > *n.Gt) &&
		(n.Gte == nil || have >= *n.Gte) && (n.Lt == nil || have < *n.Lt) &&
		(n.Lte == nil || have <= *n.Lte)
}

// capability is a compute capability as the pair the comparison orders.
type capability struct{ major, minor int }

func (a capability) cmp(b capability) int {
	switch {
	case a.major != b.major:
		return sign(a.major - b.major)
	default:
		return sign(a.minor - b.minor)
	}
}

func sign(x int) int {
	switch {
	case x < 0:
		return -1
	case x > 0:
		return 1
	}
	return 0
}

var capabilityForm = regexp.MustCompile(`^([0-9]+)\.([0-9]+)$`)

// parseCapability reads "major.minor". It refuses any other spelling, and a part too large
// for an int.
func parseCapability(s string) (capability, bool) {
	m := capabilityForm.FindStringSubmatch(strings.TrimSpace(s))
	if m == nil {
		return capability{}, false
	}
	major, err1 := strconv.Atoi(m[1])
	minor, err2 := strconv.Atoi(m[2])
	return capability{major, minor}, err1 == nil && err2 == nil
}

func (c CapabilityComparison) holds(have capability) bool {
	test := func(s *string, ok func(int) bool) bool {
		if s == nil {
			return true
		}
		want, valid := parseCapability(*s)
		return valid && ok(have.cmp(want))
	}
	return test(c.Eq, func(r int) bool { return r == 0 }) && test(c.Gt, func(r int) bool { return r > 0 }) &&
		test(c.Gte, func(r int) bool { return r >= 0 }) && test(c.Lt, func(r int) bool { return r < 0 }) &&
		test(c.Lte, func(r int) bool { return r <= 0 })
}
