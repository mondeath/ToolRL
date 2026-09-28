"""Deterministic synthetic university and integrity checks."""
import random

TABLES = ("students", "courses", "professors", "projects", "clubs", "schedules", "classrooms")

def generate_kb(seed=42):
    rng = random.Random(seed)
    kb = {table: {} for table in TABLES}
    def add(table, ident, **fields):
        kb[table][ident] = dict(id=ident, **fields)
    for i in range(5):
        add("professors", f"p{i:02}", name=f"Professor-{i:02}", department=rng.choice(["Logic", "Robotics", "Ecology"]))
    for i in range(4):
        add("classrooms", f"r{i:02}", name=f"Room-{i:02}", capacity=rng.choice([30, 40, 60]))
        add("clubs", f"b{i:02}", name=f"Club-{i:02}", members=[])
    for i in range(8):
        cid = f"c{i:02}"
        add("courses", cid, name=f"Course-{i:02}", professor=rng.choice(list(kb["professors"])), credits=rng.choice([2, 3, 4]), students=[], schedule=f"h{i:02}")
        add("schedules", f"h{i:02}", course=cid, classroom=f"r{i%4:02}", day="Monday" if i < 4 else "Tuesday", time="09:00")
    for i in range(30):
        sid = f"s{i:02}"
        courses = sorted(rng.sample(list(kb["courses"]), 3))
        clubs = sorted(rng.sample(list(kb["clubs"]), rng.randint(1, 2)))
        add("students", sid, name=f"Student-{i:02}", advisor=rng.choice(list(kb["professors"])), year=rng.randint(1, 4), courses=courses, clubs=clubs, projects=[])
        for cid in courses: kb["courses"][cid]["students"].append(sid)
        for bid in clubs: kb["clubs"][bid]["members"].append(sid)
    for i in range(5):
        pid = f"j{i:02}"
        members = sorted(rng.sample(list(kb["students"]), 6))
        add("projects", pid, name=f"Project-{i:02}", professor=rng.choice(list(kb["professors"])), members=members)
        for sid in members: kb["students"][sid]["projects"].append(pid)
    validate_kb(kb)
    return kb

RELATIONS = {
    ("students", "advisor"): "professors", ("students", "courses"): "courses",
    ("students", "clubs"): "clubs", ("students", "projects"): "projects",
    ("courses", "students"): "students", ("courses", "professor"): "professors",
    ("courses", "schedule"): "schedules", ("projects", "members"): "students",
    ("projects", "professor"): "professors", ("clubs", "members"): "students",
    ("schedules", "course"): "courses", ("schedules", "classroom"): "classrooms",
}

def validate_kb(kb):
    for (table, field), target in RELATIONS.items():
        for row in kb[table].values():
            values = row[field] if isinstance(row[field], list) else [row[field]]
            for value in values:
                if value not in kb[target]: raise ValueError(f"Dangling reference: {table}.{field}={value}")
    for table, forward, target, backward in [
        ("students", "courses", "courses", "students"),
        ("students", "clubs", "clubs", "members"),
        ("students", "projects", "projects", "members"),
        ("courses", "schedule", "schedules", "course"),
    ]:
        for a in kb[table].values():
            for b in kb[target].values():
                av = a[forward] if isinstance(a[forward], list) else [a[forward]]
                bv = b[backward] if isinstance(b[backward], list) else [b[backward]]
                if (b["id"] in av) != (a["id"] in bv): raise ValueError("Asymmetric relationship")
