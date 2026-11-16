package acceptance

import (
	"fmt"
	"math"
)

// Schedule is how a consumer realises an acceptance step by step. A profile that states none
// is realised as its mean: every request advances by the acceptance length on average, with no
// randomness, as a deterministic carry does.
type Schedule string

const (
	// ScheduleChain draws each verification step's accepted drafts as a chain over the
	// per-position rates. Draft i+1 is accepted, given that the first i were, with the
	// conditional probability p_{i+1}/p_i (p_0 = 1); the first rejection ends the chain. Each
	// request draws one random effect, a normal offset with standard deviation RequestLogitSD,
	// which it adds to the logit of every conditional probability for its whole life. The
	// stated rates are the population mean over requests: the chain's base logits are solved
	// so that averaging over the random effect gives them back (NewChain).
	ScheduleChain Schedule = "chain"
)

var schedules = map[Schedule]bool{ScheduleChain: true}

// Valid reports whether s is a stated schedule. The empty schedule, the mean, is valid too.
func (s Schedule) Valid() bool { return s == "" || schedules[s] }

// MaxRequestLogitSD bounds a chain's per-request random effect. At 5 a request one standard
// deviation out accepts almost every draft or almost none; a spread wider than that is not a
// measurement of acceptance.
const MaxRequestLogitSD = 5

// chainNodes is how many points NewChain averages the random effect over: the midpoints of
// equal-probability bins of the standard normal, z_j = Φ⁻¹((j + 0.5)/n). The same rule, with
// the same n, is what a writer that fits RequestLogitSD must use, so the fitted spread and
// the consumer's mean agree.
const chainNodes = 64

// ChainNodes returns the standard-normal points NewChain averages over, in increasing order.
func ChainNodes() []float64 {
	z := make([]float64, chainNodes)
	for j := range z {
		z[j] = math.Sqrt2 * math.Erfinv(2*(float64(j)+0.5)/chainNodes-1)
	}
	return z
}

// Chain is a chain schedule ready to draw from: the base logit of each position's conditional
// acceptance probability, and the per-request spread.
type Chain struct {
	// base[i] is the logit of position i's conditional probability for a request with no
	// random effect. +Inf is a position that is always accepted once reached, -Inf one never
	// accepted.
	base []float64
	sd   float64
}

// NewChain returns the chain schedule for per-position rates and a per-request logit spread.
// It refuses rates that are not a valid schedule (CheckRates) and a spread outside
// [0, MaxRequestLogitSD].
//
// With no spread the conditional probabilities are p_i/p_{i-1} exactly. With a spread, each
// position's base logit is the one at which the mean over the random effect of the probability
// that the first i drafts are accepted is p_i, found by bisection position by position, so the
// chain's acceptance length is the stated one.
func NewChain(rates []float64, sd float64) (*Chain, error) {
	if err := CheckRates(rates, len(rates)); err != nil {
		return nil, err
	}
	if len(rates) == 0 {
		return nil, fmt.Errorf("a chain needs at least one per-position rate")
	}
	if math.IsNaN(sd) || sd < 0 || sd > MaxRequestLogitSD {
		return nil, fmt.Errorf("request_logit_sd %v lies outside [0, %v]", sd, MaxRequestLogitSD)
	}
	c := &Chain{base: make([]float64, len(rates)), sd: sd}
	z := ChainNodes()
	// reach[j] is the probability, at node j, that every position solved so far is accepted.
	reach := make([]float64, len(z))
	for j := range reach {
		reach[j] = 1
	}
	prev := 1.0
	for i, p := range rates {
		switch {
		case p == 0:
			c.base[i] = math.Inf(-1)
		case p == prev:
			c.base[i] = math.Inf(1)
		case sd == 0:
			c.base[i] = logit(p / prev)
		default:
			c.base[i] = solveBase(reach, z, sd, p)
		}
		for j := range reach {
			reach[j] *= sigmoid(c.base[i] + sd*z[j])
		}
		prev = p
	}
	return c, nil
}

// solveBase finds the base logit b at which the mean over the nodes of reach[j]·σ(b + sd·z_j)
// is p. The mean rises with b, from 0 to the mean of reach, so bisection converges.
func solveBase(reach, z []float64, sd, p float64) float64 {
	lo, hi := -60.0, 60.0
	for range 200 {
		mid := (lo + hi) / 2
		var m float64
		for j := range z {
			m += reach[j] * sigmoid(mid+sd*z[j])
		}
		if m/float64(len(z)) < p {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// K is the number of positions the chain covers.
func (c *Chain) K() int { return len(c.base) }

// RequestLogitSD is the chain's per-request spread.
func (c *Chain) RequestLogitSD() float64 { return c.sd }

// Request returns the conditional acceptance probability of each position for one request
// whose random effect is normal·RequestLogitSD, where normal is a standard-normal draw. A
// consumer draws normal once per request and keeps the result for the request's life.
func (c *Chain) Request(normal float64) []float64 {
	out := make([]float64, len(c.base))
	for i, b := range c.base {
		out[i] = sigmoid(b + c.sd*normal)
	}
	return out
}

// DrawAccepted returns how many drafts one verification step accepts, in [0, K], for a request
// with the conditional probabilities Request returned. uniform supplies draws in [0, 1); one is
// taken per position reached, so a step draws at most K.
func DrawAccepted(conditional []float64, uniform func() float64) int {
	for i, q := range conditional {
		if uniform() >= q {
			return i
		}
	}
	return len(conditional)
}

func sigmoid(x float64) float64 {
	switch {
	case math.IsInf(x, 1):
		return 1
	case math.IsInf(x, -1):
		return 0
	}
	return 1 / (1 + math.Exp(-x))
}

func logit(q float64) float64 { return math.Log(q / (1 - q)) }
