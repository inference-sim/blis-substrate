package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// everyRunSucceeded is the check runs of a commit on which every proving job succeeded.
func everyRunSucceeded(t *testing.T, repo string) map[string]string {
	t.Helper()
	runs := map[string]string{}
	for _, names := range repoRunMap(t)[repo] {
		for _, n := range names {
			if p, ok := strings.CutSuffix(n, "*"); ok {
				n = p + "leg)"
			}
			runs[n] = "success"
		}
	}
	return runs
}

func repoRunMap(t *testing.T) RunMap {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "check-runs.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := ParseRunMap(raw)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// Every ci check of the registry has a check run that proves it, and the file names nothing
// else (LIF-3). Known-bad inputs: a registry check with no entry, and an entry for no check.
func TestEveryCICheckHasACheckRun(t *testing.T) {
	if bad := repoRunMap(t).Coverage(registry(t)); len(bad) > 0 {
		t.Fatalf("check-runs.yaml and the registry disagree:\n%s", strings.Join(bad, "\n"))
	}
	t.Run("a ci check with no entry is found", func(t *testing.T) {
		m := repoRunMap(t)
		delete(m["blis-substrate"], "TestNoNestedModule")
		has(t, m.Coverage(registry(t)), `ci check "TestNoNestedModule" has no check run`)
	})
	t.Run("an entry for no ci check is found", func(t *testing.T) {
		m := repoRunMap(t)
		m["blis-substrate"]["TestThatDoesNotExist"] = []string{"ledger"}
		has(t, m.Coverage(registry(t)), `maps "TestThatDoesNotExist"`)
	})
}

// The check-run names the file asks for are names the repositories' workflows produce. The
// substrate's are read from its own ci.yml and nightly.yml, so a renamed job fails here and not
// at a release.
func TestSubstrateRunsAreJobsOfItsWorkflows(t *testing.T) {
	names := map[string]bool{}
	for _, file := range []string{"ci.yml", "nightly.yml"} {
		raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", file))
		if err != nil {
			t.Fatal(err)
		}
		for _, line := range strings.Split(string(raw), "\n") {
			if n, ok := strings.CutPrefix(strings.TrimSpace(line), "name: "); ok && strings.HasPrefix(line, "    name:") {
				names[strings.Trim(n, `"`)] = true
			}
		}
	}
	for test, runs := range repoRunMap(t)["blis-substrate"] {
		for _, r := range runs {
			if !names[r] {
				t.Errorf("%q is proved by check run %q, which no job of ci.yml or nightly.yml is named", test, r)
			}
		}
	}
}

// A hand-written true cannot stand in for a check run (LIF-3, #15). Known-bad input: a commit
// with no check runs, and one whose proving job was skipped or failed.
func TestAHandWrittenTrueIsNotAResult(t *testing.T) {
	m := RunMap{"r": {"a test": {"unit"}, "b test": {"unit", "lint"}, "matrix": {"leg (*"}}}
	cases := []struct {
		name string
		runs map[string]string
		want map[string]bool
	}{
		{"no check runs at the commit", nil, map[string]bool{}},
		{"every proving run succeeded", map[string]string{"unit": "success", "lint": "success", "leg (a)": "success"},
			map[string]bool{"a test": true, "b test": true, "matrix": true}},
		{"one of two proving runs is missing", map[string]string{"unit": "success"}, map[string]bool{"a test": true}},
		{"a skipped job proves nothing", map[string]string{"unit": "skipped", "lint": "success"}, map[string]bool{}},
		{"a failed job is a failed check", map[string]string{"unit": "failure", "lint": "success"},
			map[string]bool{"a test": false, "b test": false}},
		{"a pending job is no result", map[string]string{"unit": "in_progress", "lint": "success"}, map[string]bool{}},
		{"a matrix with one failed leg fails", map[string]string{"leg (a)": "success", "leg (b)": "failure"}, map[string]bool{"matrix": false}},
		{"a matrix with one skipped leg is no proof", map[string]string{"leg (a)": "success", "leg (b)": "skipped"}, map[string]bool{}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := m.Results("r", c.runs)
			if len(got) != len(c.want) {
				t.Fatalf("got %v, want %v", got, c.want)
			}
			for k, v := range c.want {
				if g, ok := got[k]; !ok || g != v {
					t.Errorf("%q: got %v (present %v), want %v", k, g, ok, v)
				}
			}
		})
	}
}

// The command asks GitHub for the commit's check runs. A commit none of whose runs succeeded is
// refused by the same CheckInvariants that refused a missing result before.
func TestCommandRefusesACommitWithoutCheckRuns(t *testing.T) {
	sub := newRepo(t, subMod)
	sub.git("tag", "v0.11.0")
	h := sub.branch("b", "a.go", "package a\n", "Change")
	dir := t.TempDir()
	lock := filepath.Join(dir, "stack.lock")
	if err := os.WriteFile(lock, []byte("set: s\nmilestone: 1\nrepos:\n  - {repo: blis-substrate, branch: b, head: "+h+", bump: patch}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mapPath := filepath.Join("..", "..", ".github", "check-runs.yaml")
	old := fetchCheckRuns
	t.Cleanup(func() { fetchCheckRuns = old })

	fetchCheckRuns = func(repo, sha string) (map[string]string, error) { return map[string]string{}, nil }
	var out strings.Builder
	code, err := run([]string{"-lock", lock, "-check-runs", mapPath, "-dir", "blis-substrate=" + sub.dir}, &out)
	if err != nil {
		t.Fatal(err)
	}
	if code != 1 || !strings.Contains(out.String(), "has no recorded result") {
		t.Fatalf("a commit with no check runs was not refused: code %d\n%s", code, out.String())
	}

	// Every proving run succeeded: the same command goes ahead.
	fetchCheckRuns = func(repo, sha string) (map[string]string, error) { return everyRunSucceeded(t, repo), nil }
	out.Reset()
	code, err = run([]string{"-lock", lock, "-check-runs", mapPath, "-dir", "blis-substrate=" + sub.dir}, &out)
	if err != nil || code != 0 || !strings.Contains(out.String(), "dry run: nothing changed") {
		t.Fatalf("code %d err %v\n%s", code, err, out.String())
	}
}

// -results is gone: a file a person writes is not evidence.
func TestTheResultsFlagIsGone(t *testing.T) {
	var out strings.Builder
	code, _ := run([]string{"-lock", "x", "-results", "r.json"}, &out)
	if code != 2 {
		t.Errorf("-results was accepted (code %d)", code)
	}
}

func TestFoldRunsKeepsTheWorstConclusion(t *testing.T) {
	got := foldRuns("a\tsuccess\na\tfailure\nb\tsuccess\nb\tskipped\nc\tsuccess\n")
	if got["a"] != "failure" || got["b"] != "skipped" || got["c"] != "success" {
		t.Errorf("got %v", got)
	}
}
