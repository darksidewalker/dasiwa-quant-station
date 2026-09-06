package app

import (
 "encoding/json"
 "testing"
)

func TestLoaderMetadataRequestRoundTrip(t *testing.T) {
 for _, workflow := range []string{"quantize", "lora", "model"} {
  for _, choice := range []string{"null", "true", "false"} {
   t.Run(workflow+"/"+choice, func(t *testing.T) {
    var request any
    switch workflow {
    case "quantize": request = &QuantizeRequest{}
    case "lora": request = &LoraMergeRequest{}
    case "model": request = &ModelMergeRequest{}
    }
    body := `{}`
    if choice != "null" { body = `{"preserve_loader_metadata":`+choice+`}` }
    if err := json.Unmarshal([]byte(body), request); err != nil { t.Fatal(err) }
    encoded, err := json.Marshal(request)
    if err != nil { t.Fatal(err) }
    var payload map[string]any
    if err := json.Unmarshal(encoded, &payload); err != nil { t.Fatal(err) }
    actual, present := payload["preserve_loader_metadata"]
    if choice == "null" {
     if present { t.Fatalf("missing option must stay omitted for Python default ON: %s", encoded) }
    } else if !present || actual != (choice == "true") {
     t.Fatalf("option lost in bridge payload: %s", encoded)
    }
   })
  }
 }
}
