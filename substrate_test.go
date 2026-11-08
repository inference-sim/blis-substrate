package substrate

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"golang.org/x/mod/module"
	modzip "golang.org/x/mod/zip"
)

// The module asks for the Go its own code needs (go 1.24), not one a dependency chose: the go
// command raises a module's directive to its highest requirement's, so a pin on a module that
// declares a newer Go would silently move every consumer's toolchain (blis-latency-kernel#87).
func TestTheModuleAsksForGo124(t *testing.T) {
	raw, err := os.ReadFile("go.mod")
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(line), "go "); ok {
			if !strings.HasPrefix(v, "1.24") {
				t.Errorf("go.mod declares go %s; a dependency raised it. Keep go 1.24, and lower "+
					"the dependency's directive instead", v)
			}
			return
		}
	}
	t.Fatal("go.mod has no go directive")
}

// SUB-13: blis-substrate is one Go module. A go.mod below the root would make its directory a
// separate module, which is the split this repository exists to remove.
func TestNoNestedModule(t *testing.T) {
	for _, p := range nestedModules(t, ".") {
		t.Errorf("%s: blis-substrate is one module; remove it (SUB-13)", p)
	}
}

// nestedModules lists every go.mod other than the root's, and every go.work, under root.
func nestedModules(t *testing.T, root string) []string {
	t.Helper()
	var found []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (d.Name() == ".git" || d.Name() == "node_modules" || d.Name() == ".venv") {
			return filepath.SkipDir
		}
		rel, _ := filepath.Rel(root, path)
		if !d.IsDir() && rel != "go.mod" && (d.Name() == "go.mod" || d.Name() == "go.work") {
			found = append(found, rel)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return found
}

// X18 for SUB-13 and X11: a nested go.mod and a committed go.work are each found, and the root
// go.mod is not.
func TestNestedModuleCheckRefusesItsKnownBadInputs(t *testing.T) {
	dir := t.TempDir()
	write := func(rel string) {
		p := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("module x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod")
	if got := nestedModules(t, dir); len(got) != 0 {
		t.Fatalf("the root go.mod alone was reported: %v", got)
	}
	write("sub/pkg/go.mod")
	write("go.work")
	got := nestedModules(t, dir)
	if len(got) != 2 {
		t.Fatalf("a nested go.mod and a go.work should both be found, got %v", got)
	}
	// A module vendored under node_modules is not ours and is skipped.
	write("node_modules/dep/go.mod")
	if got := nestedModules(t, dir); len(got) != 2 {
		t.Errorf("a go.mod under node_modules was counted: %v", got)
	}
}

// SUB-13: the files in the module zip that Go fetches total under 100 MB uncompressed, well
// below Go's own limit of 500 MB. The zip is built by golang.org/x/mod/zip, the code the go
// command uses, so the files counted are exactly those a consumer downloads.
func TestModuleZipIsUnder100MB(t *testing.T) {
	var buf bytes.Buffer
	if err := modzip.CreateFromDir(&buf, module.Version{Path: "github.com/inference-sim/blis-substrate", Version: "v0.0.0"}, "."); err != nil {
		t.Fatalf("building the module zip: %v", err)
	}
	r, err := zip.NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	total := uncompressedTotal(r)
	if err := checkModuleSize(total); err != nil {
		t.Error(err)
	}
	t.Logf("module zip: %d files, %d bytes uncompressed, %d compressed", len(r.File), total, buf.Len())
}

// moduleSizeLimit is SUB-13's bound on the files a consumer downloads, uncompressed.
const moduleSizeLimit = 100 << 20

func uncompressedTotal(r *zip.Reader) uint64 {
	var total uint64
	for _, f := range r.File {
		total += f.UncompressedSize64
	}
	return total
}

func checkModuleSize(total uint64) error {
	if total >= moduleSizeLimit {
		return fmt.Errorf("the module zip holds %d bytes uncompressed; it must stay under %d (SUB-13)", total, moduleSizeLimit)
	}
	return nil
}

// X18 for SUB-13: a module of exactly the limit is refused, one byte under is accepted.
func TestModuleSizeCheckRefusesItsKnownBadInput(t *testing.T) {
	if err := checkModuleSize(moduleSizeLimit); err == nil {
		t.Error("a module of exactly 100 MiB was accepted")
	}
	if err := checkModuleSize(moduleSizeLimit + 1); err == nil {
		t.Error("a module over 100 MiB was accepted")
	}
	if err := checkModuleSize(moduleSizeLimit - 1); err != nil {
		t.Errorf("a module under the limit was refused: %v", err)
	}
}

// SUB-9: the go:embed directive must list literal patterns, so it repeats CatalogPatterns
// and RegistryPatterns. The loader reads a path with those two lists; if the directive
// differed, an embedded load and a path load would read different files.
func TestEmbedPatternsEqualLoaderPatterns(t *testing.T) {
	raw, err := os.ReadFile("data.go")
	if err != nil {
		t.Fatal(err)
	}
	if err := checkEmbedPatterns(string(raw), CatalogPatterns, RegistryPatterns); err != nil {
		t.Error(err)
	}
}

// checkEmbedPatterns compares the go:embed patterns in data.go's source with the loader's two
// lists, as sorted sets.
func checkEmbedPatterns(source string, catalog, registry []string) error {
	var embedded []string
	for _, line := range strings.Split(source, "\n") {
		if rest, ok := strings.CutPrefix(line, "//go:embed "); ok {
			embedded = append(embedded, strings.Fields(rest)...)
		}
	}
	var want []string
	for _, p := range catalog {
		want = append(want, "catalog/"+p)
	}
	for _, p := range registry {
		want = append(want, "registry/"+p)
	}
	sort.Strings(embedded)
	sort.Strings(want)
	if strings.Join(embedded, "\n") != strings.Join(want, "\n") {
		return fmt.Errorf("data.go's go:embed patterns differ from the loader's.\n go:embed: %v\n loader:   %v", embedded, want)
	}
	return nil
}

// X18 for SUB-9: a pattern embedded but not loaded, one loaded but not embedded, and a pattern
// of the wrong tree are each refused; the same set in another order is accepted.
func TestEmbedPatternCheckRefusesItsKnownBadInputs(t *testing.T) {
	src := "//go:embed catalog/a/* catalog/b/*\n//go:embed registry/c/*\nvar x int\n"
	cat, reg := []string{"a/*", "b/*"}, []string{"c/*"}
	if err := checkEmbedPatterns(src, cat, reg); err != nil {
		t.Fatalf("matching patterns were refused: %v", err)
	}
	if err := checkEmbedPatterns("//go:embed registry/c/* catalog/b/* catalog/a/*\n", cat, reg); err != nil {
		t.Errorf("the same patterns in another order were refused: %v", err)
	}
	for name, c := range map[string]struct {
		src      string
		cat, reg []string
	}{
		"embedded but not loaded":     {src, []string{"a/*"}, reg},
		"loaded but not embedded":     {src, append(append([]string{}, cat...), "z/*"), reg},
		"a pattern in the wrong tree": {src, cat, []string{"b/*"}},
		"no embed directive":          {"var x int\n", cat, reg},
	} {
		if checkEmbedPatterns(c.src, c.cat, c.reg) == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
