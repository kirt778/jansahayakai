from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    assert client.get("/api/health").json()["status"] == "ok"

def test_schemes_available():
    response = client.get("/api/schemes")
    assert response.status_code == 200
    assert response.json()["count"] >= 20

def test_eligibility_contract():
    response = client.post("/api/eligibility", json={"profile": {
        "age": 45, "gender": "male", "state": "Maharashtra",
        "occupation": "farmer", "income": 90000, "category": "OBC",
        "owns_land": True
    }})
    assert response.status_code == 200
    results = response.json()["results"]
    assert any(x["scheme_id"] == "pm_kisan" and x["status"] == "eligible" for x in results)

def test_session_and_chat():
    session = client.post("/api/session?language=en").json()
    response = client.post("/api/chat", json={
        "session_id": session["session_id"], "message": "I live in Karnataka",
        "language": "en", "profile": {"state": "Karnataka"}
    })
    assert response.status_code == 200
    assert response.json()["stage"] == "collecting_profile"
    assert response.json()["reply"]
