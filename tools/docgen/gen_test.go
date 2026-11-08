package main

import (
	"bytes"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestGeneratedDocsAreCurrent regenerates every file in memory and compares it with what is
// committed, in both directions: a file whose content changed, and a file docgen no longer
// writes (a removed vocabulary), are both stale pages a reader would trust.
func TestGeneratedDocsAreCurrent(t *testing.T) {
	root := filepath.Join("..", "..")
	files, err := generate(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, problem := range staleDocs(root, files) {
		t.Error(problem)
	}
}

// staleDocs compares generated files with what is committed under root, in both directions.
func staleDocs(root string, files []file) []string {
	var out []string
	want := map[string]bool{}
	for _, f := range files {
		want[f.path] = true
		got, err := os.ReadFile(filepath.Join(root, f.path))
		if err != nil {
			out = append(out, f.path+": not committed ("+err.Error()+"); run `go run ./tools/docgen` from the repository root")
			continue
		}
		if !bytes.Equal(got, f.content) {
			out = append(out, f.path+" is stale; run `go run ./tools/docgen` from the repository root")
		}
	}
	_ = filepath.WalkDir(filepath.Join(root, outDir), func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return nil
		}
		if !want[filepath.ToSlash(rel)] {
			out = append(out, rel+" is committed but docgen no longer writes it; run `go run ./tools/docgen`, which removes it")
		}
		return nil
	})
	return out
}

// X18 for X7: a file whose content differs, a file that is not committed, and a committed file
// the generator no longer writes are each reported; a matching tree is not.
func TestStaleDocsRefusesItsKnownBadInputs(t *testing.T) {
	root := t.TempDir()
	put := func(rel, body string) {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	a := file{path: outDir + "/a.md", content: []byte("one\n")}
	put(a.path, "one\n")
	if got := staleDocs(root, []file{a}); len(got) != 0 {
		t.Fatalf("a matching tree was reported: %v", got)
	}
	put(a.path, "two\n")
	if got := staleDocs(root, []file{a}); len(got) != 1 || !strings.Contains(got[0], "stale") {
		t.Errorf("a changed file: %v", got)
	}
	put(a.path, "one\n")
	b := file{path: outDir + "/b.md", content: []byte("x")}
	if got := staleDocs(root, []file{a, b}); len(got) != 1 || !strings.Contains(got[0], "not committed") {
		t.Errorf("an uncommitted file: %v", got)
	}
	put(outDir+"/orphan.md", "old")
	if got := staleDocs(root, []file{a}); len(got) != 1 || !strings.Contains(got[0], "no longer writes") {
		t.Errorf("an orphan: %v", got)
	}
}
