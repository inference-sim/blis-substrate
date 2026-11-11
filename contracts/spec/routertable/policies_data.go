package routertable

// The configurations below are read from llm-d's router at routerCommit. A line number is a
// line of the file at that commit.

const (
	routerRepo   = "inference-sim/llm-d-router"
	routerCommit = "b1ab0ff7c0b84090e40b874f8873e78f2ae6897c"

	pluginsDir   = "pkg/epp/framework/plugins/"
	schedDir     = pluginsDir + "scheduling/"
	producerDir  = pluginsDir + "requestcontrol/dataproducer/"
	disaggDir    = schedDir + "profilehandler/disagg/"
	pdSampleFile = "deploy/config/pd-epp-config.yaml"
	defaultsFile = "pkg/epp/config/loader/defaults.go"
)

func ev(subject, path, lines, claim string) ConfigEvidence {
	return ConfigEvidence{Subject: subject, Repository: routerRepo, Commit: routerCommit, Path: path, Lines: lines, Claim: claim}
}

func f(v float64) *float64 { return &v }

// Evidence shared by every policy: the filters, the profile handler, the decider and the picker.
func commonEvidence(picker string) []ConfigEvidence {
	pickerFile := schedDir + "picker/maxscore/picker.go"
	pickerLines := "39-40"
	if picker == "random-picker" {
		pickerFile, pickerLines = schedDir+"picker/random/picker.go", "38-39"
	}
	return []ConfigEvidence{
		ev("plugin:prefill-filter", schedDir+"filter/bylabel/role_filter.go", "30-31", "PrefillRoleType = \"prefill-filter\" selects the endpoints whose role label is prefill"),
		ev("plugin:decode-filter", schedDir+"filter/bylabel/role_filter.go", "28-29", "DecodeRoleType = \"decode-filter\" selects the endpoints whose role label is decode"),
		ev("plugin:"+picker, pickerFile, pickerLines, "the picker's registered plugin type; it takes no parameter this configuration sets (its only parameter is the optional endpoint-count cap)"),
		ev("plugin:disagg-profile-handler", disaggDir+"disagg_profile_handler.go", "47-48", "DisaggProfileHandlerType = \"disagg-profile-handler\""),
		ev("param:disagg-profile-handler.deciders.prefill", disaggDir+"disagg_profile_handler.go", "103-112",
			"deciders.prefill names the decider plugin that chooses whether a request is split; the source states no decider for a policy, and its sample (deploy/config/pd-epp-config.yaml:14-20) uses prefix-based-pd-decider. BLIS's prefill/decode deployments split every request, so the configuration names always-disagg-pd-decider"),
		ev("plugin:always-disagg-pd-decider", disaggDir+"always_disagg_pd_decider.go", "28-29", "AlwaysDisaggPDDeciderPluginType = \"always-disagg-pd-decider\""),
		ev("profile:prefill", disaggDir+"disagg_profile_handler.go", "50-51", "defaultPrefillProfile = \"prefill\": the profile the handler runs for the prefill leg"),
		ev("profile:decode", disaggDir+"disagg_profile_handler.go", "50-51", "defaultDecodeProfile = \"decode\": the profile the handler runs for the decode leg"),
	}
}

func handlerAndFilters(picker string, scorers ...ProfileRef) ([]Plugin, []Profile) {
	handler := Plugin{Type: "disagg-profile-handler", Parameters: []Parameter{
		{Name: "deciders.prefill", Value: "always-disagg-pd-decider", Origin: OriginNotStated}}}
	var plugins []Plugin
	plugins = append(plugins, Plugin{Type: "prefill-filter"}, Plugin{Type: "decode-filter"})
	profile := func(name string) Profile {
		refs := []ProfileRef{{Plugin: name + "-filter"}}
		refs = append(refs, scorers...)
		refs = append(refs, ProfileRef{Plugin: picker})
		return Profile{Name: name, Plugins: refs}
	}
	return append(plugins, handler, Plugin{Type: "always-disagg-pd-decider"}), []Profile{profile(ProfilePrefill), profile(ProfileDecode)}
}

func roundRobinConfig() PolicyConfig {
	plugins, profiles := handlerAndFilters("max-score-picker")
	plugins = append(plugins, Plugin{Type: "max-score-picker"})
	return PolicyConfig{Plugins: plugins, Profiles: profiles, Evidence: commonEvidence("max-score-picker")}
}

func randomPickerConfig() PolicyConfig {
	plugins, profiles := handlerAndFilters("random-picker")
	plugins = append(plugins, Plugin{Type: "random-picker"})
	return PolicyConfig{Plugins: plugins, Profiles: profiles, Evidence: commonEvidence("random-picker")}
}

func leastActiveConfig() PolicyConfig {
	plugins, profiles := handlerAndFilters("max-score-picker",
		ProfileRef{Plugin: "active-request-scorer", Weight: f(1), WeightOrigin: OriginSource})
	plugins = append([]Plugin{{Type: "inflight-load-producer"}}, plugins...)
	plugins = append(plugins, Plugin{Type: "active-request-scorer"}, Plugin{Type: "max-score-picker"})
	e := commonEvidence("max-score-picker")
	e = append(e,
		ev("plugin:inflight-load-producer", producerDir+"inflightload/producer.go", "49", "InFlightLoadProducerType: the producer of the router's own count of in-flight requests per endpoint"),
		ev("plugin:active-request-scorer", schedDir+"scorer/activerequest/active_request.go", "33-34", "ActiveRequestType = \"active-request-scorer\"; it reads the inflight-load-producer's count (lines 94-95, 58)"),
		ev("weight:active-request-scorer", defaultsFile, "43-44", "no file of the router sets a weight for active-request-scorer; DefaultScorerWeight = 1.0 is the weight of a scorer a configuration references without one, which is the weight used here"),
		ev("plugin:inflight-load-producer", producerDir+"inflightload/README.md", "26-32", "every parameter of the producer is optional (addEstimatedOutputTokens false, maxEstimatedOutputTokens unset, prefixMatchInfoProducerName unset); the configuration sets none, so the producer counts requests, not estimated tokens"),
	)
	return PolicyConfig{Plugins: plugins, Profiles: profiles, Evidence: e}
}

func defaultConfig() PolicyConfig {
	plugins, profiles := handlerAndFilters("max-score-picker",
		ProfileRef{Plugin: "queue-scorer", Weight: f(2), WeightOrigin: OriginSource},
		ProfileRef{Plugin: "kv-cache-utilization-scorer", Weight: f(2), WeightOrigin: OriginSource},
		ProfileRef{Plugin: "prefix-cache-scorer", Weight: f(3), WeightOrigin: OriginSource})
	producer := Plugin{Type: "approx-prefix-cache-producer", Parameters: []Parameter{
		{Name: "maxPrefixTokensToMatch", Value: "16384", Integer: true, Origin: OriginSample},
		{Name: "lruCapacityPerServer", Value: "31250", Integer: true, Origin: OriginSource},
	}}
	plugins = append([]Plugin{producer}, plugins...)
	plugins = append(plugins, Plugin{Type: "queue-scorer"}, Plugin{Type: "kv-cache-utilization-scorer"}, Plugin{Type: "prefix-cache-scorer"}, Plugin{Type: "max-score-picker"})
	e := commonEvidence("max-score-picker")
	e = append(e,
		ev("plugin:approx-prefix-cache-producer", pdSampleFile, "5-8", "the router's prefill/decode sample declares approx-prefix-cache-producer ahead of prefix-cache-scorer; the built-in default configuration (defaults.go:58-74) declares none"),
		ev("param:approx-prefix-cache-producer.maxPrefixTokensToMatch", pdSampleFile, "5-7", "the sample sets maxPrefixTokensToMatch: 16384; the router's own default is 131072 (approximateprefix/README.md:16), so 16384 is the sample's value, not the default"),
		ev("param:approx-prefix-cache-producer.lruCapacityPerServer", producerDir+"approximateprefix/README.md", "19", "lruCapacityPerServer defaults to 31250; the sample sets the same value (deploy/config/pd-epp-config.yaml:8)"),
		ev("plugin:queue-scorer", schedDir+"scorer/queuedepth/queue.go", "31", "QueueScorerType = \"queue-scorer\""),
		ev("plugin:kv-cache-utilization-scorer", schedDir+"scorer/kvcacheutilization/kvcache_utilization.go", "30", "KvCacheUtilizationScorerType = \"kv-cache-utilization-scorer\""),
		ev("plugin:prefix-cache-scorer", schedDir+"scorer/prefix/plugin.go", "63", "PrefixCacheScorerPluginType = \"prefix-cache-scorer\""),
		ev("weight:queue-scorer", defaultsFile, "49", "queueScorerWeight := 2.0, referenced with that weight by the default profile at lines 80-82"),
		ev("weight:kv-cache-utilization-scorer", defaultsFile, "50", "kvCacheUtilizationScorerWeight := 2.0, referenced with that weight by the default profile at lines 83-86"),
		ev("weight:prefix-cache-scorer", defaultsFile, "51", "prefixCacheScorerWeight := 3.0, referenced with that weight by the default profile at lines 87-90"),
	)
	return PolicyConfig{Plugins: plugins, Profiles: profiles, Evidence: e}
}
