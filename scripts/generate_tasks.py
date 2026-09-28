from _common import ROOT, write, json
from src.tasks import generate_tasks, public_task
if __name__ == "__main__":
    tasks = generate_tasks(json.loads((ROOT / "data/kb.json").read_text()))
    write("tasks.evaluation.json", tasks)
    write("tasks.public.json", [public_task(t) for t in tasks])
    print(f"Generated {len(tasks)} tasks")
