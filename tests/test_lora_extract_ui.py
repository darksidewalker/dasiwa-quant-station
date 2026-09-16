from pathlib import Path


def test_extract_ui_offers_generic_two_checkpoint_recipe():
    root = Path(__file__).resolve().parents[1]
    html = (root / "web/index.html").read_text()
    js = (root / "web/app.js").read_text()

    assert '<option value="generic" selected>Any architecture — checkpoint difference</option>' in html
    assert '<option value="h3_pruned">MiniMax H3 — pruned adapter</option>' in html
    assert 'recipe: recipe' in js
    assert 'architecture: state.architecture' in js
    assert 'recipe === "h3_pruned"' in js
    assert 'state.architecture !== "MiniMax H3"' in js
