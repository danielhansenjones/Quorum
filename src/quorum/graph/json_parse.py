from __future__ import annotations


def extract_first_json_object(raw: str) -> str | None:
    # Walk the string finding the first balanced {...} substring. Respects
    # JSON string literals so a brace inside "..." does not count, and the
    # backslash escape inside strings so an escaped quote does not break out.
    # Shared by every node that parses a JSON object out of model text; the
    # models wrap the object in prose or fences often enough that whole-string
    # json.loads is not a safe parse.
    in_string = False
    escape = False
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue
        if ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return raw[start : i + 1]
    return None
