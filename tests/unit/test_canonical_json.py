from __future__ import annotations

from quorum.cache.canonical import canonical_json, sha256_hex


def test_dict_key_order_invariance() -> None:
    a = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    b = {"content": [{"text": "hi", "type": "text"}], "role": "user"}
    assert canonical_json(a) == canonical_json(b)
    assert sha256_hex(canonical_json(a)) == sha256_hex(canonical_json(b))


def test_list_order_matters() -> None:
    # Lists in messages are semantically ordered (multi-turn). Two distinct
    # orderings must produce distinct hashes; otherwise we'd cache-collide
    # different conversations.
    a = [{"role": "user", "content": "1"}, {"role": "user", "content": "2"}]
    b = [{"role": "user", "content": "2"}, {"role": "user", "content": "1"}]
    assert canonical_json(a) != canonical_json(b)


def test_nested_dicts_sorted() -> None:
    a = {"outer": {"z": 1, "a": 2, "m": {"y": 3, "b": 4}}}
    b = {"outer": {"a": 2, "m": {"b": 4, "y": 3}, "z": 1}}
    assert canonical_json(a) == canonical_json(b)


def test_utf8_preserved() -> None:
    a = {"text": "naïve café"}
    out = canonical_json(a)
    # Non-ASCII preserved (no \u-escaping). Matches ensure_ascii=False.
    assert "naïve café".encode() in out


def test_set_normalized() -> None:
    a = {"tags": {"a", "b", "c"}}
    b = {"tags": ["a", "b", "c"]}
    assert canonical_json(a) == canonical_json(b)
