import hashlib
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
import redis.asyncio as redis

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.cache import RedisCache
from src.config import settings
from src.models import AnalysisResponse, ErrorResponse, SentimentAnalysisRequest
from src.services import SentimentAnalysisService, SentimentAnalysisError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def exception_container(app: FastAPI) -> None:
    @app.exception_handler(SentimentAnalysisError)
    async def sentiment_analysis_exception_handler(request: Request, exc: SentimentAnalysisError):
        logger.error(f"SentimentAnalysisError: {exc} - Details: {exc.details}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Sentiment analysis failed",
                details=str(exc),
                code=500
            ).model_dump(),
        )

    @app.exception_handler(TimeoutError)
    async def timeout_exception_handler(request: Request, exc: TimeoutError):
        logger.error(f"TimeoutError: {exc}")
        return JSONResponse(
            status_code=504,
            content=ErrorResponse(
                error="Analysis timeout",
                details=str(exc),
                code=504
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                details=str(exc),
                code=500
            ).model_dump(),
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.redis_client = redis.from_url(
            settings.redis.url,
            decode_responses=True,
        )

        app.state.redis_cache = RedisCache(
            redis_client=app.state.redis_client,
            ttl=settings.redis.redis_ttl,
        )

        app.state.sentiment_service = SentimentAnalysisService(
            cache=app.state.redis_cache,
            ollama_base_url=settings.ollama.base_url,
            ollama_model=settings.ollama.model,
            request_timeout=settings.ollama.request_timeout,
        )

        cache_health = await app.state.redis_cache.health_check()
        if not cache_health:
            logger.warning("Redis cache health check failed on startup")
        else:
            logger.info("Redis cache connected successfully")

    except Exception as exc:
        logger.critical(f"Failed to initialize resources on startup: {exc}")
        raise SystemExit(f"Startup failed: {exc}")

    yield

    if hasattr(app.state, "redis_client") and app.state.redis_client:
        await app.state.redis_client.aclose()
        logger.info("Redis client connection closed")

app = FastAPI(lifespan=lifespan)

exception_container(app)


def get_cache(request: Request) -> RedisCache:
    return request.app.state.redis_cache

def get_service(request: Request) -> SentimentAnalysisService:
    return request.app.state.sentiment_service

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = hashlib.md5(f"{time.time()}{request.url}".encode()).hexdigest()[:8]
    logger.info(f"[{request_id}] Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"[{request_id}] Request completed: {request.method} {request.url} - "
            f"Status: {response.status_code} - Time: {process_time:.5f}s"
        )
        return response
    except Exception as exc:
        process_time = time.time() - start_time
        logger.error(
            f"[{request_id}] Request failed: {request.method} {request.url} - "
            f"Error: {str(exc)} - Time: {process_time:.2f}s"
        )
        raise

def generate_cache_key(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_sentiment(
        request_data: SentimentAnalysisRequest,
        service: SentimentAnalysisService = Depends(get_service),
        cache: RedisCache = Depends(get_cache),
):
    start_time = time.time()
    cache_key = generate_cache_key(request_data.text)
    cached_result = await cache.get(cache_key)
    if cached_result:
        return AnalysisResponse(
            **cached_result,
            cached=True,
            processing_time=time.time() - start_time,
            timestamp=datetime.now(),
        )
    sentiment_result = await service.analyze_sentiment(request_data.text)
    response_data = {
        "text": request_data.text,
        "sentiment": sentiment_result.model_dump(),
        "model_used": settings.ollama.model,
    }
    await cache.set(cache_key, response_data)
    response_data.update({
        "cached": False,
        "processing_time": time.time() - start_time,
        "timestamp": datetime.now(),
    })
    return AnalysisResponse(**response_data)

@app.get("/health")
async def health_check(cache: RedisCache = Depends(get_cache)):
    cache_health = await cache.health_check()
    status_code = 200 if cache_health else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if cache_health else "degraded",
            "timestamp": datetime.now().isoformat(),
            "cache": "healthy" if cache_health else "unhealthy",
        }
    )

@app.post("/cache/clear")
async def clear_cache(cache: RedisCache = Depends(get_cache)):
    success = await cache.clear_cache()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear cache")
    return {"message": "Cache cleared successfully"}

@app.get("/")
async def root():
    return {
        "endpoints": {
            "analyze": "POST /analyze - Analyze text sentiment",
            "health": "GET /health - Health check",
            "cache_clear": "POST /cache/clear - Clear cache",
        },
    }
