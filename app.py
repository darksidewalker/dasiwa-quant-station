# app.py
from utils.file_ops import ensure_dirs
from ui.layout import create_ui
from ui.assets import CSS_STYLE

# 1. Initialize folders
ensure_dirs()

# 2. Build UI (Wiring is now internal to create_ui)
demo = create_ui()

# 3. Launch
if __name__ == "__main__":
    demo.launch(css=CSS_STYLE)