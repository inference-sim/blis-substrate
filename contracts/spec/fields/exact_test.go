package fields

import (
	"strings"
	"testing"
)

// listWith is a one-field list whose estimation block is est, a JSON object body.
func listWith(est string) []byte {
	return []byte(`{
  "kind": "FieldList",
  "engine": {"kind": "vllm", "version": "0.32.0"},
  "fields": [
    {"name": "VLLM_FLASHINFER_ALLREDUCE_BACKEND", "source": "env", "status": "not_modelled",
     "evidence": {"issue": "inference-sim/blis-vllm#5"},
     "estimation": ` + est + `}
  ]
}`)
}

// allReduce is the worked example: the backend FlashInfer all-reduce picks. One node: auto and
// mnnvl differ only without multicast. Several nodes: mnnvl is what auto picks on 10.0 and up.
const allReduce = `{
  "rule": "leave_unset",
  "preserves": "the backend vLLM picks for the tensor-parallel group",
  "valid_when": [
    {"value_in": ["auto"]},
    {"value_in": ["mnnvl"], "tp_group_nodes": {"eq": 1}},
    {"value_in": ["mnnvl"], "tp_group_nodes": {"gt": 1}, "compute_capability": {"gte": "10.0"}}
  ]
}`

func parseEstimation(t *testing.T, est string) Estimation {
	t.Helper()
	l, err := Parse(listWith(est))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	return l.Fields[0].Estimation
}

func nodes(n int) *int { return &n }

func TestARuleWithValidWhenLoadsAndWritesBackItsOwnBytes(t *testing.T) {
	l, err := Parse(listWith(allReduce))
	if err != nil {
		t.Fatal(err)
	}
	e := l.Fields[0].Estimation
	if e.Preserves == "" || len(e.ValidWhen) != 3 {
		t.Fatalf("estimation read as %+v", e)
	}
	out, err := l.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	again, err := Parse(out)
	if err != nil {
		t.Fatalf("the canonical bytes do not parse: %v\n%s", err, out)
	}
	if got, _ := again.Marshal(); string(got) != string(out) {
		t.Errorf("Marshal is not stable:\n%s\n%s", out, got)
	}
	if !Exact(again.Fields[0].Estimation, Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "10.3"}) {
		t.Error("the rule read back from the canonical bytes lost a clause")
	}
}

func TestEveryBadExactnessStatementIsRefused(t *testing.T) {
	const pre = `"rule": "leave_unset", "preserves": "p", `
	for _, c := range []struct {
		name, est string
		want      []string
	}{
		{"unknown clause key", `{` + pre + `"valid_when": [{"value_is": ["auto"]}]}`,
			[]string{"valid_when[0]", `unknown key "value_is"`}},
		{"unknown comparison operator", `{` + pre + `"valid_when": [{"tp_group_nodes": {"ge": 1}}]}`,
			[]string{"valid_when[0].tp_group_nodes", `unknown key "ge"`}},
		{"unknown key on the estimation", `{` + pre + `"valid_if": []}`, []string{`unknown key "valid_if"`}},
		{"capability that is not major.minor", `{` + pre + `"valid_when": [{"compute_capability": {"gte": "10"}}]}`,
			[]string{"valid_when[0].compute_capability.gte", "<major>.<minor>"}},
		{"capability that is not a number", `{` + pre + `"valid_when": [{"compute_capability": {"lt": "hopper"}}]}`,
			[]string{"compute_capability.lt"}},
		{"capability given as a number", `{` + pre + `"valid_when": [{"compute_capability": {"gte": 10.0}}]}`,
			[]string{"gte"}},
		{"node count that is not a whole number", `{` + pre + `"valid_when": [{"tp_group_nodes": {"eq": 1.5}}]}`,
			[]string{"eq"}},
		{"empty clause", `{` + pre + `"valid_when": [{}]}`, []string{"valid_when[0]", "at least one key"}},
		{"empty comparison", `{` + pre + `"valid_when": [{"tp_group_nodes": {}}]}`,
			[]string{"valid_when[0].tp_group_nodes", "at least one of eq"}},
		{"empty valid_when", `{` + pre + `"valid_when": []}`, []string{"valid_when", "at least one clause"}},
		{"empty string list", `{` + pre + `"valid_when": [{"value_in": []}]}`, []string{"value_in", "at least one string"}},
		{"blank string in a list", `{` + pre + `"valid_when": [{"value_in": ["auto", " "]}]}`, []string{"value_in[1]"}},
		{"valid_when without preserves", `{"rule": "leave_unset", "valid_when": [{"value_in": ["auto"]}]}`,
			[]string{"estimation.preserves", "required with valid_when"}},
		{"valid_when on rule none", `{"rule": "none", "valid_when": [{"value_in": ["auto"]}]}`,
			[]string{"estimation.valid_when", "only with rule"}},
		{"preserves on rule none", `{"rule": "none", "preserves": "p"}`, []string{"estimation.preserves", "only with rule"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := Parse(listWith(c.est))
			if err == nil {
				t.Fatal("Parse accepted it")
			}
			for _, w := range c.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error does not mention %q:\n%v", w, err)
				}
			}
		})
	}
}

func TestASubstituteRuleMayStateWhenItIsExact(t *testing.T) {
	e := parseEstimation(t, `{"rule": "substitute", "setting": {"kv_cache_dtype": "auto"}, "reason": "r",
	  "preserves": "p", "valid_when": [{"value_in": ["fp8"]}]}`)
	if !Exact(e, Facts{Value: "fp8"}) || Exact(e, Facts{Value: "int4"}) {
		t.Error("a substitute rule's valid_when was not applied")
	}
}

// The worked example holds only on the clauses it states.
func TestTheAllReduceStandInIsExactOnlyWhereItsClausesHold(t *testing.T) {
	e := parseEstimation(t, allReduce)
	for _, c := range []struct {
		name  string
		facts Facts
		want  bool
	}{
		{"auto needs no fact", Facts{Value: "auto"}, true},
		{"auto in other case and padding", Facts{Value: "  Auto "}, true},
		{"mnnvl on one node", Facts{Value: "mnnvl", TPGroupNodes: nodes(1)}, true},
		{"mnnvl on one node of an older part", Facts{Value: "mnnvl", TPGroupNodes: nodes(1), ComputeCapability: "9.0"}, true},
		{"mnnvl on two nodes at 10.0", Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "10.0"}, true},
		{"mnnvl on two nodes at 10.3", Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "10.3"}, true},
		{"mnnvl on two nodes at 12.0", Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "12.0"}, true},
		{"mnnvl on two nodes below 10.0", Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "9.0"}, false},
		{"capability orders by major, then minor", Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "9.10"}, false},
		{"mnnvl on two nodes, capability unknown", Facts{Value: "mnnvl", TPGroupNodes: nodes(2)}, false},
		{"mnnvl on two nodes, capability malformed", Facts{Value: "mnnvl", TPGroupNodes: nodes(2), ComputeCapability: "ten"}, false},
		{"mnnvl, node count unknown", Facts{Value: "mnnvl", ComputeCapability: "10.0"}, false},
		{"mnnvl with no facts", Facts{Value: "mnnvl"}, false},
		{"a value no clause names", Facts{Value: "trtllm", TPGroupNodes: nodes(1), ComputeCapability: "10.0"}, false},
		{"an empty value", Facts{TPGroupNodes: nodes(1)}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := Exact(e, c.facts); got != c.want {
				t.Errorf("Exact(%+v) = %v, want %v", c.facts, got, c.want)
			}
		})
	}
}

// Within a clause every key must hold; across clauses one is enough.
func TestAClauseIsAConjunctionAndTheListIsADisjunction(t *testing.T) {
	e := parseEstimation(t, `{"rule": "leave_unset", "preserves": "p", "valid_when": [
	  {"value_in": ["a"], "tp_group_nodes": {"gte": 2, "lt": 4}},
	  {"value_in": ["b"]}]}`)
	for _, c := range []struct {
		facts Facts
		want  bool
	}{
		{Facts{Value: "a", TPGroupNodes: nodes(2)}, true},
		{Facts{Value: "a", TPGroupNodes: nodes(3)}, true},
		{Facts{Value: "a", TPGroupNodes: nodes(4)}, false}, // lt fails though gte holds
		{Facts{Value: "a", TPGroupNodes: nodes(1)}, false}, // gte fails though lt holds
		{Facts{Value: "b"}, true},
		{Facts{Value: "c", TPGroupNodes: nodes(2)}, false},
	} {
		if got := Exact(e, c.facts); got != c.want {
			t.Errorf("Exact(%+v) = %v, want %v", c.facts, got, c.want)
		}
	}
}

// The UCX_TLS clause: a list that names a RDMA transport, and is not a deny list.
func TestListClausesSplitTheValueOnCommas(t *testing.T) {
	e := parseEstimation(t, `{"rule": "leave_unset", "preserves": "p", "valid_when": [
	  {"value_list_has_any": ["rc", "dc_x", "all"], "value_not_prefixed": ["^"]},
	  {"value_list_has_none": ["tcp", "cuda_ipc"], "value_in": ["sm,self"]}]}`)
	for _, c := range []struct {
		value string
		want  bool
	}{
		{"rc", true},
		{"cuda_copy, RC_X ,rc", true}, // padding and case are ignored; one match is enough
		{"dc_x,tcp", true},
		{"all", true},
		{"^rc", false},    // a deny list names rc but excludes it
		{"tcp,sm", false}, // no RDMA transport
		{"rcx", false},    // an item matches whole, not by prefix
		{"sm,self", true}, // second clause: value_in and has_none both hold
		{"", false},
	} {
		if got := Exact(e, Facts{Value: c.value}); got != c.want {
			t.Errorf("Exact(%q) = %v, want %v", c.value, got, c.want)
		}
	}
	none := parseEstimation(t, `{"rule": "leave_unset", "preserves": "p", "valid_when": [{"value_list_has_none": ["tcp"]}]}`)
	if Exact(none, Facts{Value: "sm,tcp"}) || !Exact(none, Facts{Value: "sm,self"}) || !Exact(none, Facts{Value: ""}) {
		t.Error("value_list_has_none is wrong")
	}
}

// A rule without valid_when applies to every value, as before the key existed.
func TestWithoutValidWhenTheRuleAppliesToEveryValue(t *testing.T) {
	for _, est := range []string{`{"rule": "leave_unset"}`,
		`{"rule": "leave_unset", "preserves": "the default"}`,
		`{"rule": "substitute", "setting": {"a": 1}, "reason": "r"}`} {
		e := parseEstimation(t, est)
		for _, v := range []string{"", "anything", "mnnvl"} {
			if !Exact(e, Facts{Value: v}) {
				t.Errorf("Exact(%s, %q) = false, want true", est, v)
			}
		}
	}
}

// A rule with no stand-in is never exact, whatever the value.
func TestRuleNoneIsNeverExact(t *testing.T) {
	if Exact(parseEstimation(t, `{"rule": "none"}`), Facts{Value: "auto"}) {
		t.Error("rule none reported a stand-in as exact")
	}
}
