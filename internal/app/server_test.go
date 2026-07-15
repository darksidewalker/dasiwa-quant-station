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
