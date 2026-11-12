package routertable

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestEveryPolicyHasACompleteConfiguration(t *testing.T) {
	if p := CheckPolicies(Policies()); !p.OK() {
		t.Fatalf("the policies must pass:\n%s", p.Error())
	}
}

// clone returns the policies through JSON, so a case changes its own copy.
func clone(t *testing.T) []PolicyInfo {
	t.Helper()
	b, err := json.Marshal(Policies())
	if err != nil {
		t.Fatal(err)
	}
	var out []PolicyInfo
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func byPolicy(l []PolicyInfo, p Policy) *PolicyInfo {
	for i := range l {
		if l[i].Policy == p {
			return &l[i]
		}
	}
	return nil
}

func TestKnownBadPolicyConfigurations(t *testing.T) {
	cases := []struct {
		name   string
		policy Policy
		break_ func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo
		want   string
	}{
		{"a policy removed from the list", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			var out []PolicyInfo
			for _, x := range l {
				if x.Policy != PolicyDefault {
					out = append(out, x)
				}
			}
			return out
		}, `the policy "default" of the closed list is missing`},
		{"all policies removed", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo { return nil }, `"round-robin" of the closed list is missing`},
		{"a policy outside the closed list", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			x := *in
			x.Policy = "wider"
			return append(l, x)
		}, `"wider" is not in the closed list`},
		{"a policy listed twice", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo { return append(l, *in) }, "is listed twice"},
		{"an unknown plugin", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Config.Plugins = append(in.Config.Plugins, Plugin{Type: "made-up-scorer"})
			return l
		}, `"made-up-scorer" is not a plugin type of llm-d's router`},
		{"an unknown parameter", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Config.Plugins[0].Parameters = append(in.Config.Plugins[0].Parameters, Parameter{Name: "bogus", Value: "1", Origin: OriginSource})
			return l
		}, `"bogus" is not a parameter of approx-prefix-cache-producer`},
		{"a missing required parameter", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Config.Plugins[0].Parameters = in.Config.Plugins[0].Parameters[:1]
			return l
		}, `sets no "lruCapacityPerServer"`},
		{"the producer dropped", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Config.Plugins = in.Config.Plugins[1:]
			return l
		}, "approx-prefix-cache-producer"},
		{"a missing weight", PolicyLeastActiveRequests, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			for i := range in.Config.Profiles {
				for j := range in.Config.Profiles[i].Plugins {
					in.Config.Profiles[i].Plugins[j].Weight = nil
				}
			}
			return l
		}, "a scorer needs a positive weight"},
		{"a weight that disagrees with the composition text", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			for i := range in.Config.Profiles {
				for j, r := range in.Config.Profiles[i].Plugins {
					if r.Plugin == "queue-scorer" {
						in.Config.Profiles[i].Plugins[j].Weight = f(5)
					}
				}
			}
			return l
		}, "the composition text gives queue-scorer weight 2"},
		{"the profiles disagree on a weight", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Config.Profiles[1].Plugins[1].Weight = f(7)
			return l
		}, "in another profile"},
		{"a missing profile", PolicyRoundRobin, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Config.Profiles = in.Config.Profiles[:1]
			return l
		}, "the decode profile is missing"},
		{"a profile without a picker", PolicyRoundRobin, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			p := &in.Config.Profiles[0]
			p.Plugins = p.Plugins[:len(p.Plugins)-1]
			return l
		}, "exactly one picker"},
		{"a profile with the other role's filter", PolicyRoundRobin, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Config.Profiles[0].Plugins[0].Plugin = "decode-filter"
			return l
		}, `the prefill profile filters with "prefill-filter"`},
		{"a reference to an undeclared plugin", PolicyRoundRobin, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Config.Profiles[0].Plugins = append(in.Config.Profiles[0].Plugins, ProfileRef{Plugin: "queue-scorer", Weight: f(1), WeightOrigin: OriginSource})
			return l
		}, `"queue-scorer" is not declared`},
		{"a decider that is not declared", PolicyRoundRobin, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			for i, pl := range in.Config.Plugins {
				if pl.Type == "always-disagg-pd-decider" {
					in.Config.Plugins = append(in.Config.Plugins[:i], in.Config.Plugins[i+1:]...)
					break
				}
			}
			return l
		}, "not a declared decider"},
		{"no profile handler", PolicyRoundRobin, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			var keep []Plugin
			for _, pl := range in.Config.Plugins {
				if pl.Type != "disagg-profile-handler" {
					keep = append(keep, pl)
				}
			}
			in.Config.Plugins = keep
			return l
		}, "no profile handler"},
		{"evidence removed", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			var keep []ConfigEvidence
			for _, e := range in.Config.Evidence {
				if e.Subject != "param:approx-prefix-cache-producer.maxPrefixTokensToMatch" {
					keep = append(keep, e)
				}
			}
			in.Config.Evidence = keep
			return l
		}, `"param:approx-prefix-cache-producer.maxPrefixTokensToMatch" has no evidence`},
		{"evidence of something the configuration lacks", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Config.Evidence = append(in.Config.Evidence, ev("plugin:queue-scorer-2", "a.go", "1", "x"))
			return l
		}, "is no plugin, parameter, weight or profile"},
		{"a cited value without lines", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			for i := range in.Config.Evidence {
				if in.Config.Evidence[i].Subject == "weight:queue-scorer" {
					in.Config.Evidence[i].Lines = ""
				}
			}
			return l
		}, "so the passage is cited at its lines"},
		{"evidence of a repository that is not the router's", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Config.Evidence[0].Repository = "someone/else"
			return l
		}, "must be a repository of llm-d's router"},
		{"an unpinned commit", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Config.Evidence[0].Commit = "main"
			return l
		}, "full 40-character"},
		{"an origin outside the three", PolicyDefault, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Config.Plugins[0].Parameters[0].Origin = "guess"
			return l
		}, "must be \"source\", \"sample\" or \"not-stated\""},
		{"the composition text names a plugin the configuration lacks", PolicyRoundRobin, func(l []PolicyInfo, in *PolicyInfo) []PolicyInfo {
			in.Plugins = "queue-scorer (1); max-score-picker"
			return l
		}, `names "queue-scorer", which the configuration does not declare`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			l := clone(t)
			in := byPolicy(l, c.policy)
			l = c.break_(l, in)
			p := CheckPolicies(l)
			if p.OK() {
				t.Fatalf("accepted a bad list")
			}
			if !strings.Contains(p.Error(), c.want) {
				t.Errorf("want a problem containing %q, got:\n%s", c.want, p.Error())
			}
		})
	}
}

// The one value that the source states differently from the value used must say so.
func TestADeploymentSampleValueNamesTheRoutersDefault(t *testing.T) {
	d := byPolicy(Policies(), PolicyDefault)
	for _, e := range d.Config.Evidence {
		if e.Subject == "param:approx-prefix-cache-producer.maxPrefixTokensToMatch" {
			if !strings.Contains(e.Claim, "131072") {
				t.Errorf("the sample's 16384 must name the router's default 131072: %q", e.Claim)
			}
			return
		}
	}
	t.Fatal("no evidence for maxPrefixTokensToMatch")
}

func TestTheConfigurationRoundTripsAsJSON(t *testing.T) {
	b, err := json.Marshal(Policies())
	if err != nil {
		t.Fatal(err)
	}
	var back []PolicyInfo
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if p := CheckPolicies(back); !p.OK() {
		t.Fatalf("the JSON form must carry everything the check needs:\n%s", p.Error())
	}
	for _, key := range []string{`"scheduling_profiles"`, `"weight_origin"`, `"deciders.prefill"`, `"maxPrefixTokensToMatch"`} {
		if !strings.Contains(string(b), key) {
			t.Errorf("the JSON lacks %s", key)
		}
	}
}
