# infra/redis_config.py
"""
Redis connection factory and configuration.

Provides a centralized way to create Redis clients with consistent
configuration across the pipeline modules.
"""

import redis
import config


def get_redis_client() -> redis.Redis:
    """
    Create and return a Redis client using global config.

    Returns
    -------
    redis.Redis
        Connected Redis client instance.
    """
    return redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
        retry_on_timeout=True,
    )


def check_redis_connection() -> bool:
    """
    Test Redis connectivity.

    Returns
    -------
    bool
        True if Redis is reachable.
    """
    try:
        client = get_redis_client()
        client.ping()
        return True
    except (redis.ConnectionError, redis.TimeoutError):
        return False
