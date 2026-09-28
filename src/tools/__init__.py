"""KB-only queries. No task or answer access."""
import json
from src.kb import RELATIONS

class QueryEngine:
    def __init__(self, kb):
        self.kb = kb

    def path(self, table, row, path, evidence):
        field, *rest = path.split(".")
        value = row[field]
        evidence.append({"entity": row["id"], "field": field, "value": value})
        values = value if isinstance(value, list) else [value]
        if not rest: return values
        target = RELATIONS[(table, field)]
        return [v for ident in values for v in self.path(target, self.kb[target][ident], ".".join(rest), evidence)]

    def execute(self, query, limit=None, shallow=False):
        table = query["table"]
        select = query.get("select", "id")
        where = query.get("where", [])
        if shallow and any("." in p for p in [select] + [c["field"] for c in where]):
            raise ValueError("search supports only single-table fields")
        if query.get("reduce", "list") not in ("list", "scalar"): raise ValueError("Invalid reduce")
        results, evidence = [], []
        for ident, row in sorted(self.kb[table].items()):
            matched = True
            for cond in where:
                values = self.path(table, row, cond["field"], evidence)
                op, expected = cond.get("op", "eq"), cond["value"]
                if op == "eq": ok = expected in values
                elif op == "gte": ok = any(v >= expected for v in values)
                elif op == "contains": ok = any(str(expected).casefold() in str(v).casefold() for v in values)
                else: raise ValueError(f"Unsupported operator: {op}")
                if not ok:
                    matched = False
                    break
            if matched:
                results.append({"entity": ident, "values": self.path(table, row, select, evidence)})
        total = len(results)
        if limit is not None: results = results[:limit]
        # Unique values retain deterministic traversal order; list tasks use set semantics.
        values = list(dict.fromkeys(v for row in results for v in row["values"]))
        if query.get("reduce") == "scalar":
            if len(values) != 1: raise ValueError("Scalar query must produce exactly one unique value")
            answer = values[0]
        else: answer = values
        unique = {json.dumps(e, sort_keys=True): e for e in evidence}
        return {"answer": answer, "results": results, "evidence": list(unique.values()), "truncated": total > len(results), "matched_count": total}

    def lookup(self, entity, field):
        matches = [rows[entity] for rows in self.kb.values() if entity in rows]
        if len(matches) != 1: raise ValueError("Unknown or ambiguous entity")
        value = matches[0][field]
        return {"value": value, "evidence": [{"entity": entity, "field": field, "value": value}]}

    def search(self, query, top_k):
        if isinstance(query, dict):
            result = self.execute(query, limit=top_k, shallow=True)
            # search exposes retrieved rows only, not evidence from filtered-out rows.
            result.pop("evidence")
            return result
        if not isinstance(query, str) or not query.strip(): raise ValueError("Nonempty keyword required")
        hits = [{"table": table, "entity": ident, "record": row}
                for table, rows in self.kb.items() for ident, row in sorted(rows.items())
                if query.casefold() in json.dumps(row).casefold()]
        return {"results": hits[:top_k], "truncated": len(hits) > top_k, "matched_count": len(hits)}
