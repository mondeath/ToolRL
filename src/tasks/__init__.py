"""Evaluation records and a strictly smaller public task view."""
from src.tools import QueryEngine

def public_task(task):
    return {"task_id": task["task_id"], "question": task["question"]}

def generate_tasks(kb):
    tasks = []
    engine = QueryEngine(kb)
    def add(kind, question, query):
        result = engine.execute(query)
        tasks.append(dict(task_id=f"task-{len(tasks):03}", question=question,
                          task_type=kind, ground_truth=result["answer"],
                          evaluation={"query_plan": query, "underlying_facts": result["evidence"]}))
    for i in range(16):
        sid = f"s{i:02}"
        add("single-hop", f"What is the advisor ID of student {sid}?", {"table":"students", "where":[{"field":"id", "value":sid}], "select":"advisor", "reduce":"scalar"})
        add("two-hop", f"What is the department of the advisor of student {sid}?", {"table":"students", "where":[{"field":"id", "value":sid}], "select":"advisor.department", "reduce":"scalar"})
    for year in range(1, 5):
        add("list/filter", f"List student IDs in year {year}.", {"table":"students", "where":[{"field":"year", "value":year}]})
        for p in range(5):
            add("multi-condition", f"List student IDs in year {year} with advisor p{p:02}.", {"table":"students", "where":[{"field":"year", "value":year}, {"field":"advisor", "value":f"p{p:02}"}]})
    for c in range(8):
        add("list/filter", f"List student IDs enrolled in course c{c:02}.", {"table":"students", "where":[{"field":"courses", "value":f"c{c:02}"}]})
    return tasks
