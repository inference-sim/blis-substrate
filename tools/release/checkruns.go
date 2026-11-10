package main

import (
	"bytes"
	"fmt"
	"os/exec"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

// RunMap says which check runs prove each registry check (.github/check-runs.yaml): for a
// repository, a check's test string to the check-run names that must all have concluded
// success at the commit. A name ending in "*" matches every run that begins with the text
// before it, and at least one must exist (a matrix).
type RunMap map[string]map[string][]string

// ParseRunMap reads the mapping strictly: an unknown shape is an error.
func ParseRunMap(raw []byte) (RunMap, error) {
	var m RunMap
	if err := validate.DecodeStrict("check-runs.yaml", raw, &m); err != nil {
		return nil, err
	}
	for repo, tests := range m {
		for test, names := range tests {
			if len(names) == 0 {
				return nil, fmt.Errorf("check-runs.yaml: %s: %q names no check run", repo, test)
			}
		}
	}
	return m, nil
}

// Coverage lists the disagreements between the mapping and the registry: a ci check with no
// entry, and an entry that names no ci check. The mapping is data to keep, so a drift in either
// direction fails a test and not a release.
func (m RunMap) Coverage(reg *invariants.Registry) []string {
	want := map[string]map[string]bool{}
	for _, inv := range reg.Invariants {
		for _, c := range inv.Checks {
			if c.Status == invariants.StatusCI && c.Runs != invariants.RunsPractice {
				if want[c.Repo] == nil {
					want[c.Repo] = map[string]bool{}
				}
				want[c.Repo][c.Test] = true
			}
		}
	}
	var bad []string
	for repo, tests := range want {
		for test := range tests {
			if _, ok := m[repo][test]; !ok {
				bad = append(bad, fmt.Sprintf("%s: ci check %q has no check run in check-runs.yaml", repo, test))
			}
		}
	}
	for repo, tests := range m {
		for test := range tests {
			if !want[repo][test] {
				bad = append(bad, fmt.Sprintf("%s: check-runs.yaml maps %q, which is no ci check of the registry", repo, test))
			}
		}
	}
	sort.Strings(bad)
	return bad
}

// Results builds, for one repository at one commit, each ci check's test string and whether the
// check runs that prove it all succeeded. runs maps a check-run name to its conclusion at that
// commit, with a name that ran more than once ("success" only if every run did). A check whose
// proof is absent, or only skipped or pending, has no entry: it has no recorded result (LIF-3).
// A named run that exists and did not succeed is false. Nothing here reads a value a person
// wrote.
func (m RunMap) Results(repo string, runs map[string]string) map[string]bool {
	out := map[string]bool{}
	for test, names := range m[repo] {
		proved, failed := true, false
		for _, n := range names {
			matched := 0
			for have, concl := range runs {
				if !matchesRun(n, have) {
					continue
				}
				matched++
				switch concl {
				case "success":
				case "failure", "cancelled", "timed_out", "action_required", "startup_failure", "stale":
					failed = true
				default: // skipped, neutral, pending, in progress: not proof
					proved = false
				}
			}
			if matched == 0 {
				proved = false
			}
		}
		switch {
		case failed:
			out[test] = false
		case proved:
			out[test] = true
		}
	}
	return out
}

func matchesRun(pattern, name string) bool {
	if p, ok := strings.CutSuffix(pattern, "*"); ok {
		return strings.HasPrefix(name, p)
	}
	return name == pattern
}

// FetchCheckRuns returns, for a commit of an inference-sim repository, each check-run name and
// its conclusion ("success" only if every run of that name succeeded; the status when a run has
// not concluded). It asks the GitHub API through the gh command.
func FetchCheckRuns(repo, sha string) (map[string]string, error) {
	cmd := exec.Command("gh", "api", "--paginate", "-H", "Accept: application/vnd.github+json",
		fmt.Sprintf("repos/inference-sim/%s/commits/%s/check-runs?per_page=100", repo, sha),
		"--jq", `.check_runs[] | [.name, (.conclusion // .status)] | @tsv`)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("check runs of %s at %s: %v: %s", repo, sha, err, strings.TrimSpace(stderr.String()))
	}
	return foldRuns(string(out)), nil
}

// foldRuns reads "name<TAB>conclusion" lines. A name that appears more than once keeps the
// worst conclusion, so one failed run of a name is not hidden by another run of it that passed.
func foldRuns(tsv string) map[string]string {
	rank := func(c string) int {
		switch c {
		case "success":
			return 0
		case "skipped", "neutral":
			return 1
		case "queued", "in_progress", "pending", "waiting", "requested":
			return 2
		}
		return 3
	}
	m := map[string]string{}
	for _, line := range strings.Split(strings.TrimSpace(tsv), "\n") {
		name, concl, ok := strings.Cut(line, "\t")
		if !ok {
			continue
		}
		if prev, seen := m[name]; !seen || rank(concl) > rank(prev) {
			m[name] = concl
		}
	}
	return m
}
