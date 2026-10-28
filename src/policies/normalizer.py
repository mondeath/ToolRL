"""Level-1 surface normalization only; callers must strictly validate the result."""
from dataclasses import dataclass
import json
import re
from typing import Optional
from .parser import _pairs


@dataclass(frozen=True)
class NormalizationResult:
    raw_output: str
    normalization_applied: bool = False
    normalization_type: Optional[str] = None
    normalized_output: Optional[str] = None


def normalize_protocol_output(raw_output):
    """Wrap prefixed arguments or remove a matching redundant envelope prefix.

    Keep the JSON payload verbatim: no argument reserialization or schema repair.
    This helper does not establish validity; parse_action remains authoritative.
    """
    unchanged = NormalizationResult(raw_output)
    if not isinstance(raw_output, str):
        return unchanged
    match = re.fullmatch(
        r'\s*(lookup|search|aggregate|deep_search|verify|submit|direct_answer)(?:\s*:\s*|\s+)(\{.*\})\s*',
        raw_output, flags=re.DOTALL)
    if not match:
        return unchanged
    action, payload = match.groups()
    try:
        value = json.loads(payload, object_pairs_hook=_pairs)
    except (ValueError, RecursionError):
        return unchanged
    if not isinstance(value, dict):
        return unchanged
    if 'action' in value or 'arguments' in value:
        if set(value) != {'action', 'arguments'} or value['action'] != action:
            return unchanged
        kind, normalized = 'redundant_action_prefix', payload
    else:
        kind = 'action_prefix_arguments'
        normalized = '{"action":' + json.dumps(action) + ',"arguments":' + payload + '}'
    return NormalizationResult(raw_output, True, kind, normalized)
