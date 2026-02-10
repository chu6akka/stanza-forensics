import json

from app.core.metrics.pos import pos_profile
from app.core.backends.interface import Token


def test_reproducible_json_without_timestamp():
    toks = [
        Token("Он", "он", "PRON", {}, 0, 2),
        Token("пишет", "писать", "VERB", {}, 3, 8),
    ]
    payload = {"backend": "pymorphy3", "metrics": pos_profile(toks), "sha": "abc"}
    a = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    b = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert a == b
