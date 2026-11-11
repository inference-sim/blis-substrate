package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/routertable"
)

// The dashboard reads this output, so its shape is pinned: the closed list, and for each policy
// the plugins, the profiles with weights, and the evidence.
func TestRouterPoliciesPublishesTheCheckedConfigurations(t *testing.T) {
	var out bytes.Buffer
	if err := run([]string{"router-policies"}, &out); err != nil {
		t.Fatal(err)
	}
	var got struct {
		Closed   []string `json:"closed_policies"`
		Policies []struct {
			Policy string `json:"policy"`
			Config struct {
				Plugins  []struct{ Type string } `json:"plugins"`
				Profiles []struct {
					Name    string
					Plugins []struct {
						Plugin string
						Weight *float64
					}
				} `json:"scheduling_profiles"`
				Evidence []struct{ Subject, Commit, Path, Lines string } `json:"evidence"`
			} `json:"config"`
		} `json:"policies"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Closed) != len(routertable.ClosedPolicies) || len(got.Policies) != len(routertable.ClosedPolicies) {
		t.Fatalf("want one configuration per closed policy %v, got %v / %d", routertable.ClosedPolicies, got.Closed, len(got.Policies))
	}
	for _, p := range got.Policies {
		if len(p.Config.Plugins) == 0 || len(p.Config.Profiles) != 2 || len(p.Config.Evidence) == 0 {
			t.Errorf("%s: incomplete output %+v", p.Policy, p.Config)
		}
	}
}
