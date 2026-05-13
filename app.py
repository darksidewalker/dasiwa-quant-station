# app.py
from utils.file_ops import ensure_dirs
from ui.layout import create_ui
from ui.assets import CSS_STYLE, get_theme

# 1. Initialize folders
ensure_dirs()

# 2. Build UI (wiring is internal to create_ui)
demo = create_ui()

# 3. Launch. In Gradio 6, theme and css must be passed here, not on
# gr.Blocks() in layout.py.
if __name__ == "__main__":
    demo.launch(theme=get_theme(), css=CSS_STYLE)
