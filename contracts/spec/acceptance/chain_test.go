package acceptance

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// stepsAccepting draws n steps for one request and returns, for each i in 0..K, the fraction of
// steps that accepted at least i+1 drafts, and the mean drafts accepted.
func stepsAccepting(conditional []float64, n int, rng *rand.Rand) ([]float64, float64) {
	atLeast := make([]float64, len(conditional))
	var total float64
	for range n {
		a := DrawAccepted(conditional, rng.Float64)
		total += float64(a)
		for i := range a {
			atLeast[i]++
		}
	}
	for i := range atLeast {
		atLeast[i] /= float64(n)
	}
	return atLeast, total / float64(n)
}

// With no spread, a chain accepts the first i drafts with exactly the stated probability p_i, so
// a step accepts AL-1 drafts on average.
func TestAChainWithNoSpreadReproducesTheStatedRates(t *testing.T) {
	rates := []float64{0.84, 0.70, 0.59}
	c, err := NewChain(rates, 0)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	atLeast, mean := stepsAccepting(c.Request(rng.NormFloat64()), 400_000, rng)
	for i, p := range rates {
		if math.Abs(atLeast[i]-p) > 0.004 {
			t.Errorf("P(first %d accepted) = %.4f, want %.2f", i+1, atLeast[i], p)
		}
	}
	if math.Abs(mean-sum(rates)) > 0.01 {
		t.Errorf("mean drafts accepted %.4f, want %.2f", mean, sum(rates))
	}
}

// With a spread, requests differ from one another, but averaged over requests the chain still
// accepts the stated rates: the stated rates are the population's, not one request's.
func TestASpreadKeepsThePopulationRatesAndSpreadsRequests(t *testing.T) {
	rates := []float64{0.84, 0.70, 0.59}
	perRequestSD := func(sd float64) (atLeast []float64, spread float64) {
		c, err := NewChain(rates, sd)
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewPCG(7, 11))
		atLeast = make([]float64, len(rates))
		const requests, steps = 4000, 200
		var means []float64
		for range requests {
			a, m := stepsAccepting(c.Request(rng.NormFloat64()), steps, rng)
			for i := range a {
				atLeast[i] += a[i] / requests
			}
			means = append(means, m)
		}
		return atLeast, stddev(means)
	}
	flatAt, flat := perRequestSD(0)
	wideAt, wide := perRequestSD(0.6)
	for i, p := range rates {
		if math.Abs(wideAt[i]-p) > 0.01 {
			t.Errorf("with a spread, P(first %d accepted) over requests = %.4f, want %.2f", i+1, wideAt[i], p)
		}
		if math.Abs(flatAt[i]-p) > 0.01 {
			t.Errorf("with no spread, P(first %d accepted) over requests = %.4f, want %.2f", i+1, flatAt[i], p)
		}
	}
	// 200 steps alone leave a spread of about sqrt(var/200); a logit spread of 0.6 adds more
	// than doubles it.
	if wide < 2*flat {
		t.Errorf("per-request spread of mean accepted drafts: %.3f with request_logit_sd 0.6, %.3f with none; want the first at least twice the second", wide, flat)
	}
}

// A position whose rate equals the one before it is always accepted once reached, and one whose
// rate is zero is never reached, whatever the spread.
func TestAChainKeepsCertainPositionsCertain(t *testing.T) {
	c, err := NewChain([]float64{1, 1, 0.12, 0}, 0.8)
	if err != nil {
		t.Fatal(err)
	}
	for _, normal := range []float64{-3, 0, 3} {
		q := c.Request(normal)
		if q[0] != 1 || q[1] != 1 || q[3] != 0 {
			t.Errorf("at normal %v the conditional probabilities are %v; want 1, 1, then 0 at the last", normal, q)
		}
	}
	// The minimum-variance schedule as a chain with no spread is the engine's synthetic one.
	mv, err := MinimumVarianceRates(3.12, 3)
	if err != nil {
		t.Fatal(err)
	}
	c, err = NewChain(mv, 0)
	if err != nil {
		t.Fatal(err)
	}
	if q := c.Request(0); q[0] != 1 || q[1] != 1 || math.Abs(q[2]-0.12) > 1e-12 {
		t.Errorf("minimum-variance rates %v as a chain: %v", mv, q)
	}
}

func TestNewChainRefuses(t *testing.T) {
	for _, c := range []struct {
		name  string
		rates []float64
		sd    float64
		want  string
	}{
		{"no rates", nil, 0, "at least one"},
		{"a rising rate", []float64{0.5, 0.6}, 0, "exceeds"},
		{"a rate above 1", []float64{1.2}, 0, "outside [0, 1]"},
		{"a negative spread", []float64{0.8}, -0.1, "request_logit_sd"},
		{"a NaN spread", []float64{0.8}, math.NaN(), "request_logit_sd"},
		{"a spread past the bound", []float64{0.8}, MaxRequestLogitSD + 1, "request_logit_sd"},
	} {
		if _, err := NewChain(c.rates, c.sd); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want one containing %q", c.name, err, c.want)
		}
	}
}

func chainProfile(mut ...func(*Profile)) Profile {
	p := Profile{Model: "m", SpeculativeMethod: "mtp", NumSpecTokens: 3, Thinking: ThinkingOn,
		Category: "agentic", AcceptanceLength: 3.13, PerPositionRates: []float64{0.84, 0.70, 0.59},
		Schedule: ScheduleChain, RequestLogitSD: 0.4, Method: vocab.MethodMeasured,
		Provenance: &Provenance{Source: "s", Hardware: "h200", EngineImage: "e",
			Sampling: &Sampling{Temperature: fp(1)}}}
	for _, m := range mut {
		m(&p)
	}
	return p
}

func TestAProfileStatesItsScheduleCorrectly(t *testing.T) {
	if p := set("s", chainProfile()).Validate(); !p.OK() {
		t.Fatalf("a valid chain profile is refused: %s", p.Error())
	}
	if p := set("s", chainProfile(func(p *Profile) { p.Schedule, p.RequestLogitSD = "", 0 })).Validate(); !p.OK() {
		t.Fatalf("a profile with no schedule is refused: %s", p.Error())
	}
	for _, c := range []struct {
		name string
		mut  func(*Profile)
		at   string
	}{
		{"an unknown schedule", func(p *Profile) { p.Schedule = "bernoulli" }, "profiles[0].schedule"},
		{"a chain with no stated rates", func(p *Profile) { p.PerPositionRates = nil }, "profiles[0].schedule"},
		{"a spread with the mean schedule", func(p *Profile) { p.Schedule = "" }, "profiles[0].request_logit_sd"},
		{"a negative spread", func(p *Profile) { p.RequestLogitSD = -0.1 }, "profiles[0].request_logit_sd"},
		{"a spread past the bound", func(p *Profile) { p.RequestLogitSD = 6 }, "profiles[0].request_logit_sd"},
		{"an infinite spread", func(p *Profile) { p.RequestLogitSD = math.Inf(1) }, "profiles[0].request_logit_sd"},
	} {
		p := set("s", chainProfile(c.mut)).Validate()
		if p.OK() || !strings.Contains(p.Error(), c.at) {
			t.Errorf("%s: got %q, want a problem at %s", c.name, p.Error(), c.at)
		}
	}
}

// An exact-key profile hands its schedule to the run; a value borrowed from it at another K, and
// a value stated inline, carry the mean.
func TestTheScheduleTravelsOnlyWithAnExactKey(t *testing.T) {
	s := set("s", chainProfile())
	r, err := Resolve(Request{Model: "m", SpeculativeMethod: "mtp", NumSpecTokens: 3, Registry: []*ProfileSet{s},
		Workload: Workload{Category: "agentic", Thinking: ThinkingOn}})
	if err != nil {
		t.Fatal(err)
	}
	if r.Schedule != ScheduleChain || r.RequestLogitSD != 0.4 {
		t.Errorf("exact key: schedule %q sd %v, want chain 0.4", r.Schedule, r.RequestLogitSD)
	}
	if c, err := r.Chain(); err != nil || c == nil || c.K() != 3 || c.RequestLogitSD() != 0.4 {
		t.Errorf("exact key: chain %v, %v", c, err)
	}

	r, err = Resolve(Request{Model: "m", SpeculativeMethod: "mtp", NumSpecTokens: 2, Registry: []*ProfileSet{s},
		Workload: Workload{Category: "agentic", Thinking: ThinkingOn}})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Borrowed() || r.Schedule != "" || r.RequestLogitSD != 0 {
		t.Errorf("borrowed at another K: schedule %q sd %v, want the mean", r.Schedule, r.RequestLogitSD)
	}
	if c, err := r.Chain(); c != nil || err != nil {
		t.Errorf("borrowed at another K: chain %v, %v; want none", c, err)
	}

	al := 3.13
	r, err = Resolve(Request{Model: "m", SpeculativeMethod: "mtp", NumSpecTokens: 3, Registry: []*ProfileSet{s},
		Workload: Workload{Category: "agentic", Thinking: ThinkingOn},
		CLI:      &Value{NumSpecTokens: 3, AcceptanceLength: &al}})
	if err != nil {
		t.Fatal(err)
	}
	if r.Schedule != "" {
		t.Errorf("a stated value carries schedule %q, want the mean", r.Schedule)
	}
}

func stddev(xs []float64) float64 {
	var m float64
	for _, x := range xs {
		m += x
	}
	m /= float64(len(xs))
	var v float64
	for _, x := range xs {
		v += (x - m) * (x - m)
	}
	return math.Sqrt(v / float64(len(xs)))
}
