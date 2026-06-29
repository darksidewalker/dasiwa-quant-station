package pathbrowser

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Item struct {
	Name  string `json:"name"`
	Path  string `json:"path"`
	IsDir bool   `json:"is_dir"`
}

type Response struct {
	Path   string `json:"path"`
	Parent string `json:"parent"`
	Items  []Item `json:"items"`
}

func Browse(path string, filterDirs bool) (*Response, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}

	items := make([]Item, 0)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if filterDirs && !info.IsDir() {
			continue
		}
		items = append(items, Item{
			Name:  e.Name(),
			Path:  filepath.Join(path, e.Name()),
			IsDir: info.IsDir(),
		})
	}

	sort.Slice(items, func(i, j int) bool {
		if items[i].IsDir != items[j].IsDir {
			return items[i].IsDir
		}
		return strings.ToLower(items[i].Name) < strings.ToLower(items[j].Name)
	})

	parent := filepath.Dir(path)
	return &Response{
		Path:   path,
		Parent: parent,
		Items:  items,
	}, nil
}
