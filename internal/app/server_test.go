package app

import (
	"context"
	"path/filepath"
	"reflect"
	"testing"
)

func TestUpdateStepsPullsLatestSourceBeforeSetupAndBuild(t *testing.T) {
	root := "/tmp/dasiwa-quant-station"
	steps := updateSteps(context.Background(), root)

	if len(steps) != 3 {
		t.Fatalf("expected 3 update steps, got %d", len(steps))
	}

	if steps[0].name != "source update" {
		t.Fatalf("first step = %q, want source update", steps[0].name)
	}
	if got, want := steps[0].cmd.Args, []string{"git", "pull", "--ff-only"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("source update command = %q, want %q", got, want)
	}
	if steps[0].cmd.Dir != root {
		t.Fatalf("source update directory = %q, want %q", steps[0].cmd.Dir, root)
	}

	if steps[1].name != "setup" || steps[1].cmd.Args[1] != filepath.Join(root, "start-linux.sh") || steps[1].cmd.Args[2] != "--setup-only" {
		t.Fatalf("second step should run setup from %q, got %#v", root, steps[1].cmd)
	}
	if steps[2].name != "build" || steps[2].cmd.Args[1] != "build" {
		t.Fatalf("third step should build the Go app, got %#v", steps[2].cmd)
	}
}

func TestFormatSupportedFor(t *testing.T) {
	cases := []struct {
		format string
		arch   string
		want   bool
	}{
		{"W4A8", "MiniMax H3", true},
		{"W4A8", "LTX-2.3", false},
		{"W4A8", "WAN 2.2", false},
		{"INT4 ConvRot Runtime", "Krea 2", true},
		{"INT4 ConvRot Runtime", "Flux.2", false},
		{"INT4 ConvRot Runtime", "Not set", false},
		{"FP8", "Any Arch", true}, // unrestricted format
		{"GGUF_Q4_K", "Any Arch", true},
	}
	for _, c := range cases {
		if got := formatSupportedFor(c.format, c.arch); got != c.want {
			t.Errorf("formatSupportedFor(%q, %q) = %v, want %v", c.format, c.arch, got, c.want)
		}
	}
}
