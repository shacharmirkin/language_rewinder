from pathlib import Path
import py_compile
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_config_loads():
    import yaml
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "models" in cfg and "main" in cfg["models"]

def test_app_compiles():
    py_compile.compile(str(ROOT / "app.py"), doraise=True)
