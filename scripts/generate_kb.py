from _common import ROOT, write, json
from src.kb import generate_kb
if __name__ == "__main__":
    config = json.loads((ROOT / "configs/default.json").read_text())
    write("kb.json", generate_kb(config["seed"]))
    print("Generated data/kb.json")
