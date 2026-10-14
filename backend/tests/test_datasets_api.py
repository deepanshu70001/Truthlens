import pytest


@pytest.mark.asyncio
async def test_list_datasets_in_demo_mode(client, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")

    response = await client.get("/api/v1/datasets")

    assert response.status_code == 200
    data = response.json()
    assert data["data_status"] == "SYNTHETIC_DEMO_ONLY"
    assert data["is_synthetic"] is True
    assert data["is_real_world_observation"] is False

    names = {item["name"] for item in data["datasets"]}
    assert names == {"virality", "regional_spread", "social_impact"}
    assert all(item["available"] for item in data["datasets"])


@pytest.mark.asyncio
async def test_list_datasets_blocked_without_demo_mode(client, monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)

    response = await client.get("/api/v1/datasets")

    assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dataset_name",
    ["virality", "regional_spread", "social_impact"],
)
async def test_read_dataset_pagination(client, monkeypatch, dataset_name):
    monkeypatch.setenv("DEMO_MODE", "true")

    response = await client.get(
        f"/api/v1/datasets/{dataset_name}",
        params={"limit": 5, "offset": 0},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == dataset_name
    assert data["data_status"] == "SYNTHETIC_DEMO_ONLY"
    assert data["total_records"] == 500
    assert len(data["records"]) == 5
    assert data["is_synthetic"] is True
    assert data["is_real_world_observation"] is False
    assert data["score_interpretation"] == "simulated_only_not_a_prediction"


@pytest.mark.asyncio
async def test_read_dataset_blocked_without_demo_mode(client, monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)

    response = await client.get("/api/v1/datasets/virality")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_unknown_dataset_returns_404(client, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")

    response = await client.get("/api/v1/datasets/unknown_dataset")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_invalid_pagination_returns_422(client, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")

    response = await client.get(
        "/api/v1/datasets/virality",
        params={"limit": 0},
    )

    assert response.status_code == 422
