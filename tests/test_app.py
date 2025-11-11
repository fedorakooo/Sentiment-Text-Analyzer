import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock

from src.main import app, get_cache, get_service
from src.models import SentimentResult, SentimentLabel
from src.exceptions import SentimentAnalysisError


@pytest.fixture
def mock_cache():
    return AsyncMock()


@pytest.fixture
def mock_service():
    return AsyncMock()


@pytest_asyncio.fixture
async def client(mock_cache, mock_service):
    app.dependency_overrides[get_cache] = lambda: mock_cache
    app.dependency_overrides[get_service] = lambda: mock_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        await app.router.startup()
        yield ac
        await app.router.shutdown()

    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_analyze_sentiment_success_cache_miss(client, mock_cache, mock_service):
    mock_cache.get.return_value = None
    mock_service.analyze_sentiment.return_value = SentimentResult(
        label=SentimentLabel.POSITIVE,
        confidence=0.95,
        explanation="Great product!"
    )
    mock_cache.set.return_value = True

    request_data = {"text": "I love this product!"}

    response = await client.post("/analyze", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == request_data["text"]
    assert data["cached"] is False
    assert data["sentiment"]["label"] == "positive"
    assert data["sentiment"]["confidence"] == 0.95

    mock_cache.get.assert_called_once()
    mock_service.analyze_sentiment.assert_called_once_with(request_data["text"])
    mock_cache.set.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_sentiment_success_cache_hit(client, mock_cache, mock_service):
    cached_data = {
        "text": "I love this product!",
        "sentiment": {"label": "positive", "confidence": 0.88, "explanation": "Cached explanation"},
        "model_used": "llama2"
    }
    mock_cache.get.return_value = cached_data

    request_data = {"text": "I love this product!"}

    response = await client.post("/analyze", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is True
    assert data["sentiment"]["label"] == "positive"
    assert data["model_used"] == "llama2"

    mock_cache.get.assert_called_once()
    mock_service.analyze_sentiment.assert_not_called()
    mock_cache.set.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_sentiment_service_error(client, mock_cache, mock_service):
    mock_cache.get.return_value = None
    mock_service.analyze_sentiment.side_effect = SentimentAnalysisError("LLM is down", details={"code": "LLM_FAIL"})

    response = await client.post("/analyze", json={"text": "This will fail"})

    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "Sentiment analysis failed"
    assert data["details"] == "LLM is down"


@pytest.mark.asyncio
async def test_analyze_sentiment_timeout_error(client, mock_cache, mock_service):
    mock_cache.get.return_value = None
    mock_service.analyze_sentiment.side_effect = TimeoutError("Request timed out after 30s")

    response = await client.post("/analyze", json={"text": "This will time out"})

    assert response.status_code == 504
    data = response.json()
    assert data["error"] == "Analysis timeout"
    assert "timed out" in data["details"]


@pytest.mark.asyncio
async def test_health_check_healthy(client, mock_cache):
    mock_cache.health_check.return_value = True
    response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["cache"] == "healthy"


@pytest.mark.asyncio
async def test_health_check_degraded(client, mock_cache):
    mock_cache.health_check.return_value = False
    response = await client.get("/health")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["cache"] == "unhealthy"


@pytest.mark.asyncio
async def test_clear_cache_success(client, mock_cache):
    mock_cache.clear_cache.return_value = True
    response = await client.post("/cache/clear")

    assert response.status_code == 200
    assert response.json() == {"message": "Cache cleared successfully"}


@pytest.mark.asyncio
async def test_clear_cache_failure(client, mock_cache):
    mock_cache.clear_cache.return_value = False
    response = await client.post("/cache/clear")

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to clear cache"}
