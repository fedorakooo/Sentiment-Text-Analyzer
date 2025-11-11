import json
import logging
from typing import Any
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, redis_client: redis.Redis, ttl: int = 3600, cache_enabled: bool = True):
        self.redis = redis_client
        self.ttl = ttl
        self.cache_enabled = cache_enabled

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get value from Redis"""
        if not self.cache_enabled:
            return None
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as exc:
            logger.error(f"Redis get error: {str(exc)}")
            return None

    async def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> bool:
        """Set value in Redis"""
        if not self.cache_enabled:
            return False
        try:
            actual_ttl = ttl or self.ttl
            serialized_value = json.dumps(value, default=str)
            await self.redis.setex(key, actual_ttl, serialized_value)
            return True
        except Exception as exc:
            logger.error(f"Redis set error: {str(exc)}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        try:
            result = await self.redis.delete(key)
            return bool(result)
        except Exception as exc:
            logger.error(f"Redis delete error: {str(exc)}")
            return False

    async def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            await self.redis.ping()
            return True
        except Exception as exc:
            logger.error(f"Redis health check failed: {str(exc)}")
            return False

    async def clear_cache(self) -> bool:
        """Clear all cache entries"""
        try:
            await self.redis.flushdb()
            return True
        except Exception as exc:
            logger.error(f"Cache clear error: {str(exc)}")
            return False
