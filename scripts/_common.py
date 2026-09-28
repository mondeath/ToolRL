import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def write(name, value):
    path = ROOT / "data" / name
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
