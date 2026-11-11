package routertable

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	validate "github.com/inference-sim/blis-substrate/contracts/check"
)

// PolicyConfig is the complete endpoint-picker (EPP) configuration of one policy in a
// prefill/decode deployment: every plugin instance with its parameters, the scheduling profiles
// with their plugin references and weights, and the passage of llm-d's router that supports
// each value. A consumer that writes an EPP file renders it and defines nothing of its own.
//
// Every value carries an Origin. A value the router's source states is cited at file and
// line; a value the source does not state is marked OriginNotStated with a claim that says
// so, and is the choice BLIS's prefill/decode deployments make, not a claim about the router.
type PolicyConfig struct {
	// Plugins are the plugin instances, in declaration order. The name of an instance is its
	// Type: a configuration holds at most one instance of a type.
	Plugins []Plugin `json:"plugins"`
	// Profiles are the scheduling profiles. A prefill/decode deployment has exactly two,
	// "prefill" and "decode".
	Profiles []Profile `json:"scheduling_profiles"`
	// Evidence supports the plugins, parameters and weights, by subject (see Subject*).
	Evidence []ConfigEvidence `json:"evidence"`
}

// Plugin is one plugin instance.
type Plugin struct {
	// Type is the plugin type of llm-d's router, one of KnownPlugins.
	Type string `json:"type"`
	// Parameters are the parameters the configuration sets. A parameter left out takes the
	// router's own default. A nested parameter has a dotted name, "deciders.prefill".
	Parameters []Parameter `json:"parameters,omitempty"`
}

// Parameter is one parameter of a plugin.
type Parameter struct {
	// Name is the parameter's name in the router's configuration, dotted when nested.
	Name string `json:"name"`
	// Value is the value as text. Integer says it is written as a YAML integer, not a string.
	Value   string `json:"value"`
	Integer bool   `json:"integer,omitempty"`
	// Origin says where the value comes from.
	Origin Origin `json:"origin"`
}

// Profile is one scheduling profile.
type Profile struct {
	// Name is "prefill" or "decode".
	Name string `json:"name"`
	// Plugins are the references in order: filters, scorers, picker.
	Plugins []ProfileRef `json:"plugins"`
}

// ProfileRef is a reference to a plugin instance from a profile.
type ProfileRef struct {
	// Plugin is the Type of an instance in PolicyConfig.Plugins.
	Plugin string `json:"plugin"`
	// Weight is set for a scorer and only for a scorer.
	Weight *float64 `json:"weight,omitempty"`
	// WeightOrigin says where the weight comes from; set exactly when Weight is.
	WeightOrigin Origin `json:"weight_origin,omitempty"`
}

// Origin says what supports a value of a policy's configuration.
type Origin string

const (
	// OriginSource: the router's code or documentation states the value, as its default or as
	// the policy's own setting, at the cited lines.
	OriginSource Origin = "source"
	// OriginSample: a sample deployment file of the router sets the value, at the cited lines.
	// The router's default is another value; the claim says which.
	OriginSample Origin = "sample"
	// OriginNotStated: the router's source does not state the value for this policy. The
	// value is what BLIS's prefill/decode deployments use, and the claim says what the source
	// does show.
	OriginNotStated Origin = "not-stated"
)

// ConfigEvidence is one passage of llm-d's router that supports one subject of a
// PolicyConfig.
type ConfigEvidence struct {
	// Subject names what the passage supports: "plugin:<type>", "param:<type>.<name>",
	// "weight:<type>" or "profile:<name>".
	Subject    string `json:"subject"`
	Repository string `json:"repository"`
	Commit     string `json:"commit"`
	Path       string `json:"path"`
	// Lines is a line or a range "<first>-<last>". Required unless the subject's origin is
	// OriginNotStated.
	Lines string `json:"lines,omitempty"`
	// Claim says what the passage shows, and for OriginNotStated what it does not.
	Claim string `json:"claim"`
}

// PluginKind is the part a plugin type plays in a configuration.
type PluginKind string

const (
	KindProducer       PluginKind = "producer"
	KindFilter         PluginKind = "filter"
	KindScorer         PluginKind = "scorer"
	KindPicker         PluginKind = "picker"
	KindProfileHandler PluginKind = "profile-handler"
	KindDecider        PluginKind = "decider"
)

// PluginSpec is what a configuration may say of a known plugin type.
type PluginSpec struct {
	Kind PluginKind
	// Parameters are the parameter names the type accepts.
	Parameters []string
	// Required are the parameters a complete configuration of the type sets.
	Required []string
}

// KnownPlugins are the plugin types a policy's configuration may use, with what the router's
// source accepts of each. The list grows with the router code a policy needs.
var KnownPlugins = map[string]PluginSpec{
	"approx-prefix-cache-producer": {Kind: KindProducer,
		Parameters: []string{"autoTune", "blockSizeTokens", "maxPrefixBlocksToMatch", "maxPrefixTokensToMatch", "lruCapacityPerServer"},
		Required:   []string{"maxPrefixTokensToMatch", "lruCapacityPerServer"}},
	"inflight-load-producer": {Kind: KindProducer,
		Parameters: []string{"addEstimatedOutputTokens", "maxEstimatedOutputTokens", "prefixMatchInfoProducerName"}},
	"prefill-filter":              {Kind: KindFilter},
	"decode-filter":               {Kind: KindFilter},
	"active-request-scorer":       {Kind: KindScorer, Parameters: []string{"idleThreshold", "maxBusyScore", "inFlightLoadProducerName"}},
	"queue-scorer":                {Kind: KindScorer},
	"kv-cache-utilization-scorer": {Kind: KindScorer},
	"prefix-cache-scorer":         {Kind: KindScorer, Parameters: []string{"prefixMatchInfoProducerName", "matchLengthWeight", "matchLengthScaleTokens"}},
	"max-score-picker":            {Kind: KindPicker},
	"random-picker":               {Kind: KindPicker},
	"disagg-profile-handler":      {Kind: KindProfileHandler, Parameters: []string{"stageOrder", "deciders.prefill", "deciders.encode"}, Required: []string{"deciders.prefill"}},
	"always-disagg-pd-decider":    {Kind: KindDecider},
	"prefix-based-pd-decider":     {Kind: KindDecider, Parameters: []string{"nonCachedTokens"}},
}

// ClosedPolicies are the policies the table offers, stated apart from Policies() so that a
// policy dropped from Policies() is a failure of CheckPolicies and not a silent shrinking of
// the list the consumers iterate. Adding a policy is a change here and in Policies().
var ClosedPolicies = []Policy{PolicyRoundRobin, PolicyRandomPicker, PolicyLeastActiveRequests, PolicyDefault}

// Profile names of a prefill/decode deployment.
const (
	ProfilePrefill = "prefill"
	ProfileDecode  = "decode"
)

// CheckPolicies checks a policy list as a whole: it is exactly ClosedPolicies, each policy
// once, and each carries a complete configuration. It returns every problem, or nil. The
// loader of the table and the tests of consumers call it on Policies().
func CheckPolicies(infos []PolicyInfo) *validate.Problems {
	p := &validate.Problems{}
	have := map[Policy]bool{}
	for i, in := range infos {
		at := fmt.Sprintf("policies[%d]", i)
		if have[in.Policy] {
			p.Field(at+".policy", "%q is listed twice", in.Policy)
		}
		have[in.Policy] = true
		closed := false
		for _, c := range ClosedPolicies {
			closed = closed || c == in.Policy
		}
		if !closed {
			p.Field(at+".policy", "%q is not in the closed list %v: a policy is added with its configuration and in ClosedPolicies", in.Policy, ClosedPolicies)
		}
		in.Config.validate(p, at+"("+string(in.Policy)+").config", in)
	}
	for _, c := range ClosedPolicies {
		if !have[c] {
			p.Field("policies", "the policy %q of the closed list is missing: removing a policy removes a configuration consumers select", c)
		}
	}
	return p
}

func (c PolicyConfig) validate(p *validate.Problems, at string, in PolicyInfo) {
	byType := map[string]Plugin{}
	subjects := map[string]Origin{}
	for i, pl := range c.Plugins {
		pa := fmt.Sprintf("%s.plugins[%d]", at, i)
		spec, ok := KnownPlugins[pl.Type]
		switch {
		case pl.Type == "":
			p.Field(pa+".type", "required")
			continue
		case !ok:
			p.Field(pa+".type", "%q is not a plugin type of llm-d's router this table knows: %v", pl.Type, knownPluginNames())
			continue
		}
		if _, dup := byType[pl.Type]; dup {
			p.Field(pa+".type", "%q is declared twice: an instance is named by its type", pl.Type)
		}
		byType[pl.Type] = pl
		subjects["plugin:"+pl.Type] = ""
		set := map[string]bool{}
		for j, pr := range pl.Parameters {
			pp := fmt.Sprintf("%s.parameters[%d]", pa, j)
			switch {
			case !contains(spec.Parameters, pr.Name):
				p.Field(pp+".name", "%q is not a parameter of %s (it accepts %v)", pr.Name, pl.Type, spec.Parameters)
			case set[pr.Name]:
				p.Field(pp+".name", "%q is set twice", pr.Name)
			}
			set[pr.Name] = true
			if pr.Value == "" {
				p.Field(pp+".value", "required")
			}
			if pr.Integer {
				if _, err := strconv.Atoi(pr.Value); err != nil {
					p.Field(pp+".value", "is marked an integer but is %q", pr.Value)
				}
			}
			if !pr.Origin.valid() {
				p.Field(pp+".origin", "must be %q, %q or %q, got %q", OriginSource, OriginSample, OriginNotStated, pr.Origin)
			}
			subjects["param:"+pl.Type+"."+pr.Name] = pr.Origin
		}
		for _, r := range spec.Required {
			if !set[r] {
				p.Field(pa+".parameters", "%s is incomplete: it sets no %q, and a consumer would have to choose the value", pl.Type, r)
			}
		}
	}
	// The deciders the profile handler names must be declared.
	if h, ok := firstOfKind(c, KindProfileHandler); !ok {
		p.Field(at+".plugins", "no profile handler: a prefill/decode configuration needs disagg-profile-handler")
	} else {
		for _, pr := range h.Parameters {
			if strings.HasPrefix(pr.Name, "deciders.") {
				if d, ok := byType[pr.Value]; !ok || KnownPlugins[d.Type].Kind != KindDecider {
					p.Field(at+".plugins", "%s.%s names %q, which is not a declared decider", h.Type, pr.Name, pr.Value)
				}
			}
		}
	}
	// Profiles.
	names := map[string]bool{}
	weights := map[string]float64{}
	for i, pf := range c.Profiles {
		pa := fmt.Sprintf("%s.scheduling_profiles[%d]", at, i)
		if pf.Name != ProfilePrefill && pf.Name != ProfileDecode {
			p.Field(pa+".name", "must be %q or %q, got %q", ProfilePrefill, ProfileDecode, pf.Name)
		}
		if names[pf.Name] {
			p.Field(pa+".name", "%q is listed twice", pf.Name)
		}
		names[pf.Name] = true
		subjects["profile:"+pf.Name] = ""
		filters, pickers := 0, 0
		for j, ref := range pf.Plugins {
			ra := fmt.Sprintf("%s.plugins[%d]", pa, j)
			pl, ok := byType[ref.Plugin]
			if !ok {
				p.Field(ra+".plugin", "%q is not declared in plugins", ref.Plugin)
				continue
			}
			switch KnownPlugins[pl.Type].Kind {
			case KindFilter:
				filters++
				if want := pf.Name + "-filter"; pl.Type != want {
					p.Field(ra+".plugin", "the %s profile filters with %q, got %q", pf.Name, want, pl.Type)
				}
			case KindPicker:
				pickers++
			case KindScorer:
				if ref.Weight == nil || *ref.Weight <= 0 {
					p.Field(ra+".weight", "a scorer needs a positive weight, and the configuration states it")
					continue
				}
				if !ref.WeightOrigin.valid() {
					p.Field(ra+".weight_origin", "required: %q, %q or %q", OriginSource, OriginSample, OriginNotStated)
				}
				if w, seen := weights[pl.Type]; seen && w != *ref.Weight {
					p.Field(ra+".weight", "%s has weight %v here and %v in another profile: the policy states one", pl.Type, *ref.Weight, w)
				}
				weights[pl.Type] = *ref.Weight
				subjects["weight:"+pl.Type] = ref.WeightOrigin
			default:
				p.Field(ra+".plugin", "%s is a %s, which a profile does not reference", pl.Type, KnownPlugins[pl.Type].Kind)
			}
			if ref.Weight != nil && KnownPlugins[pl.Type].Kind != KindScorer {
				p.Field(ra+".weight", "only a scorer has a weight")
			}
		}
		if filters != 1 {
			p.Field(pa+".plugins", "the %s profile needs exactly one filter, has %d", pf.Name, filters)
		}
		if pickers != 1 {
			p.Field(pa+".plugins", "the %s profile needs exactly one picker, has %d", pf.Name, pickers)
		}
	}
	for _, n := range []string{ProfilePrefill, ProfileDecode} {
		if !names[n] {
			p.Field(at+".scheduling_profiles", "the %s profile is missing", n)
		}
	}
	// Every declared plugin is referenced by a profile, or is a producer or the handler or a
	// decider (which profiles do not reference).
	for t, pl := range byType {
		switch KnownPlugins[t].Kind {
		case KindFilter, KindScorer, KindPicker:
			if _, ok := weights[t]; ok {
				continue
			}
			used := false
			for _, pf := range c.Profiles {
				for _, r := range pf.Plugins {
					used = used || r.Plugin == pl.Type
				}
			}
			if !used {
				p.Field(at+".plugins", "%s is declared but no profile references it", t)
			}
		}
	}
	// The scorers and weights agree with the composition text the table states.
	c.checkAgainstText(p, at, in, weights, byType)
	// Evidence.
	c.checkEvidence(p, at, subjects)
}

// checkAgainstText compares the configuration with the composition text of the policy
// ("queue-scorer (2); ...; max-score-picker"), so the two statements cannot drift.
func (c PolicyConfig) checkAgainstText(p *validate.Problems, at string, in PolicyInfo, weights map[string]float64, byType map[string]Plugin) {
	text := map[string]string{}
	for _, item := range strings.Split(in.Plugins, ";") {
		item = strings.TrimSpace(item)
		if item == "" || strings.HasPrefix(item, "no scorer") {
			continue
		}
		name, w, _ := strings.Cut(item, " (")
		text[name] = strings.TrimSuffix(w, ")")
	}
	for name, w := range text {
		pl, ok := byType[name]
		if !ok {
			p.Field(at+".plugins", "the composition text names %q, which the configuration does not declare", name)
			continue
		}
		if w != "" && KnownPlugins[pl.Type].Kind == KindScorer && fmt.Sprint(weights[name]) != w {
			p.Field(at+".scheduling_profiles", "the composition text gives %s weight %s, the configuration %v", name, w, weights[name])
		}
	}
	for t, pl := range byType {
		switch KnownPlugins[pl.Type].Kind {
		case KindProducer, KindScorer, KindPicker:
			if _, ok := text[t]; !ok {
				p.Field(at+".plugins", "the configuration declares %s, which the composition text %q does not name", t, in.Plugins)
			}
		}
	}
}

func (c PolicyConfig) checkEvidence(p *validate.Problems, at string, subjects map[string]Origin) {
	cited := map[string]bool{}
	for i, ev := range c.Evidence {
		ea := fmt.Sprintf("%s.evidence[%d]", at, i)
		if _, ok := subjects[ev.Subject]; !ok {
			p.Field(ea+".subject", "%q is no plugin, parameter, weight or profile of this configuration (subjects: %v)", ev.Subject, sortedKeys(subjects))
			continue
		}
		if !contains(LLMDRepositories, ev.Repository) {
			p.Field(ea+".repository", "must be a repository of llm-d's router %v, got %q", LLMDRepositories, ev.Repository)
		}
		if !commit.MatchString(ev.Commit) {
			p.Field(ea+".commit", "must be a full 40-character lowercase commit, got %q", ev.Commit)
		}
		if strings.TrimSpace(ev.Path) == "" || strings.HasPrefix(ev.Path, "/") {
			p.Field(ea+".path", "required: a file from the repository's root")
		}
		if strings.TrimSpace(ev.Claim) == "" {
			p.Field(ea+".claim", "required: what the passage shows")
		}
		if ev.Lines != "" {
			m := lines.FindStringSubmatch(ev.Lines)
			if m == nil {
				p.Field(ea+".lines", "must be a line or a range \"<first>-<last>\", got %q", ev.Lines)
			} else if m[2] != "" {
				a, _ := strconv.Atoi(m[1])
				b, _ := strconv.Atoi(m[2])
				if b < a {
					p.Field(ea+".lines", "the range %q ends before it starts", ev.Lines)
				}
			}
		}
		if subjects[ev.Subject] != OriginNotStated && ev.Lines == "" {
			p.Field(ea+".lines", "required: the subject's origin is %q, so the passage is cited at its lines", subjects[ev.Subject])
		}
		cited[ev.Subject] = true
	}
	for s := range subjects {
		if !cited[s] {
			p.Field(at+".evidence", "%q has no evidence: cite the passage of llm-d's router that supports it, or one that shows the router does not state it", s)
		}
	}
}

func (o Origin) valid() bool { return o == OriginSource || o == OriginSample || o == OriginNotStated }

func firstOfKind(c PolicyConfig, k PluginKind) (Plugin, bool) {
	for _, pl := range c.Plugins {
		if KnownPlugins[pl.Type].Kind == k {
			return pl, true
		}
	}
	return Plugin{}, false
}

func contains(l []string, s string) bool {
	for _, x := range l {
		if x == s {
			return true
		}
	}
	return false
}

func knownPluginNames() []string { return sortedKeysSpec(KnownPlugins) }

func sortedKeysSpec(m map[string]PluginSpec) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortedKeys(m map[string]Origin) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
