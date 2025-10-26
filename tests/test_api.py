# tests/test_api.py
import requests
import os

API = "http://127.0.0.1:8000"

def test_health():
    r = requests.get(f"{API}/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
