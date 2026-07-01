"""
API smoke test — exercises the no-LLM paths (no API key required):
health, UI route, out-of-scope streaming, and feedback logging.
"""
from fastapi.testclient import TestClient
import api

with TestClient(api.app) as client:
    # health
    h = client.get("/health").json()
    print("health:", h)
    assert h["knowledge_base_ready"] is True

    # UI served
    r = client.get("/")
    print("root content-type:", r.headers["content-type"], "| status:", r.status_code)
    assert "Acme Corp Assistant" in r.text

    # streaming, out-of-scope (short-circuits before the LLM)
    print("\nstreaming out-of-scope query:")
    with client.stream("POST", "/query/stream",
                       json={"query": "what is the company lunch subsidy"}) as s:
        for line in s.iter_lines():
            if line:
                print("  ", line)

    # feedback
    fb = client.post("/feedback", json={
        "query": "test q", "response": "test a", "rating": "down",
        "comment": "smoke test"}).json()
    print("\nfeedback:", fb)

print("\nAll no-LLM API paths OK.")
