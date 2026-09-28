"""Strict semantic validation; only an entire optional JSON code fence is unwrapped."""
import json
import math
import re
from dataclasses import dataclass
from typing import Optional
from .schema import FIELDS, RELATIONS


@dataclass
class ParseResult:
    success: bool
    parsed_action: Optional[dict] = None
    error: Optional[str] = None
    invalid_arguments: bool = False


def _scalar(value):
    return value is None or isinstance(value, (str, bool, int)) or (isinstance(value, float) and math.isfinite(value))


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _path(table, path, shallow):
    if not isinstance(path, str) or not path or (shallow and '.' in path):
        raise ValueError('Invalid field path')
    parts = path.split('.')
    for i, part in enumerate(parts):
        if part not in FIELDS[table]: raise ValueError('Unknown field for table: ' + part)
        if i < len(parts)-1:
            if part not in RELATIONS.get(table, {}): raise ValueError('Cannot traverse a non-relation field')
            table = RELATIONS[table][part]


def _query(query, shallow):
    if not isinstance(query, dict) or not set(query) <= {'table', 'where', 'select', 'reduce'}:
        raise ValueError('query must be an object with table/where/select/reduce only')
    table = query.get('table')
    if not isinstance(table, str) or table not in FIELDS: raise ValueError('Unknown table')
    _path(table, query.get('select', 'id'), shallow)
    if query.get('reduce', 'list') not in ('list', 'scalar'): raise ValueError('Invalid reduce')
    conditions = query.get('where', [])
    if not isinstance(conditions, list): raise ValueError('where must be an array')
    for condition in conditions:
        if (not isinstance(condition, dict) or not {'field', 'value'} <= set(condition)
                or not set(condition) <= {'field', 'op', 'value'}):
            raise ValueError('Invalid where condition keys')
        _path(table, condition['field'], shallow)
        if condition.get('op', 'eq') not in ('eq', 'gte', 'contains'): raise ValueError('Invalid operator')
        if not _scalar(condition['value']): raise ValueError('Condition value must be a scalar')


def parse_action(raw):
    if not isinstance(raw, str): return ParseResult(False, error='Output must be text')
    text = raw.strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, flags=re.DOTALL)
    if fence: text = fence.group(1).strip()
    try:
        value = json.loads(text, object_pairs_hook=_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON number')))
    except (ValueError, RecursionError) as exc:
        return ParseResult(False, error='Invalid JSON: ' + str(exc))
    if not isinstance(value, dict) or set(value) != {'action', 'arguments'}:
        return ParseResult(False, error='Expected exactly action and arguments')
    action, arguments = value['action'], value['arguments']
    if not isinstance(action, str) or action not in ('lookup', 'search', 'deep_search', 'direct_answer'):
        return ParseResult(False, parsed_action=value, error='Unknown action')
    try:
        if not isinstance(arguments, dict): raise ValueError('arguments must be an object')
        required = {'lookup': {'entity', 'field'}, 'search': {'query'}, 'deep_search': {'query'}, 'direct_answer': {'final_answer'}}[action]
        if set(arguments) != required: raise ValueError('Missing or extra arguments')
        if action == 'lookup':
            if any(not isinstance(v, str) or not v.strip() for v in arguments.values()): raise ValueError('lookup arguments must be nonempty strings')
            if '.' in arguments['field'] or arguments['field'] not in {f for fs in FIELDS.values() for f in fs}:
                raise ValueError('lookup requires one known field')
        elif action in ('search', 'deep_search'):
            query = arguments['query']
            if action == 'search' and isinstance(query, str):
                if not query.strip(): raise ValueError('Empty keyword query')
            else: _query(query, shallow=action == 'search')
        else:
            answer = arguments['final_answer']
            if not (_scalar(answer) or isinstance(answer, list) and all(_scalar(v) for v in answer)):
                raise ValueError('final_answer must be a scalar or array of scalars')
    except (ValueError, TypeError, RecursionError) as exc:
        return ParseResult(False, parsed_action=value, error=str(exc), invalid_arguments=True)
    return ParseResult(True, parsed_action=value)
