"""Endpoint tests for the Flask web layer (simple_web_app)."""

import pytest

import simple_web_app


@pytest.fixture
def client():  # type: ignore[no-untyped-def]
    simple_web_app.app.config.update(TESTING=True)
    return simple_web_app.app.test_client()


def test_index_serves(client) -> None:  # type: ignore[no-untyped-def]
    assert client.get("/").status_code == 200


def test_simulate_single_soliton(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.post(
        "/api/simulate",
        json={
            "N": 128,
            "L": 50.0,
            "dt": 0.01,
            "t_final": 2.0,
            "solitons": [{"kappa": 0.5, "x0": 25}],
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert len(data["solutions"]) == len(data["times"])
    assert {"mass", "momentum", "energy"} <= set(data["invariants"])
    assert len(data["mode_energy"]) == 6
    assert "predicted_solitons" in data
    assert "auto_resolution" in data


def test_missing_parameter_returns_400(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.post("/api/simulate", json={"L": 50.0, "dt": 0.01, "t_final": 2.0})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_gaussian_predicts_multiple_solitons(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.post(
        "/api/simulate",
        json={
            "N": 256,
            "L": 100.0,
            "dt": 0.01,
            "t_final": 1.0,
            "gaussians": [{"amplitude": 1.0, "x0": 50, "width": 3.0}],
        },
    )
    assert resp.status_code == 200
    assert len(resp.get_json()["predicted_solitons"]) >= 2
