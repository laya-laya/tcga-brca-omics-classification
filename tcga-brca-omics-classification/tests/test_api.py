import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from tcga_brca.api import app  # noqa: E402


@pytest.fixture
def client(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    with TestClient(app) as c:  # runs the startup hook that loads the model
        yield c


def _payload(X):
    return {"samples": {s: row.to_dict() for s, row in X.astype(float).iterrows()}}


def test_health_and_model_card(client):
    assert client.get("/health").json()["status"] == "ok"
    card = client.get("/model").json()
    assert card["classes"] == ["Basal", "Her2", "LumA", "LumB"]


def test_predict_returns_one_prediction_per_sample(client, batch):
    response = client.post("/predict", json=_payload(batch.iloc[:5]))
    assert response.status_code == 200
    body = response.json()
    assert [p["sample_id"] for p in body["predictions"]] == list(batch.index[:5])
    for p in body["predictions"]:
        assert np.isclose(sum(p["probabilities"].values()), 1.0, atol=1e-4)
    assert {"validation", "drift", "summary", "model"} <= set(body)


def test_predict_rejects_samples_without_model_genes(client):
    response = client.post("/predict", json={"samples": {"S1": {"FOO": 1.0}}})
    assert response.status_code == 422
    assert "missing" in response.json()["detail"]


def test_health_is_503_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
