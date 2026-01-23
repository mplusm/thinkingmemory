import json
from datetime import timedelta
from typing import Optional

from thinkingmemory.core.database import get_redis


def _get_client():
    """Get the Redis client instance."""
    return get_redis()

def store_working_memory(agent_id: str, key: str, value: dict, ttl: int = 300) -> bool:
    """
    Store a key-value pair in working memory with TTL.

    Args:
        agent_id: The agent's unique identifier
        key: The key to store the value under
        value: The dictionary value to store
        ttl: Time-to-live in seconds (default: 300 = 5 minutes)

    Returns:
        True if stored successfully
    """
    full_key = f"{agent_id}:{key}"
    _get_client().setex(full_key, timedelta(seconds=ttl), json.dumps(value))
    return True

def retrieve_working_memory(agent_id: str, key: str) -> Optional[dict]:
    """
    Retrieve a value from working memory.

    Args:
        agent_id: The agent's unique identifier
        key: The key to retrieve

    Returns:
        The stored dictionary or None if not found
    """
    full_key = f"{agent_id}:{key}"
    value = _get_client().get(full_key)
    return json.loads(value) if value else None

def delete_working_memory(agent_id: str, key: str) -> bool:
    """
    Delete a specific key from working memory.

    Args:
        agent_id: The agent's unique identifier
        key: The key to delete

    Returns:
        True if deleted, False if key didn't exist
    """
    full_key = f"{agent_id}:{key}"
    return _get_client().delete(full_key) > 0

def list_working_memory_keys(agent_id: str) -> list[str]:
    """
    List all keys in working memory for an agent.

    Args:
        agent_id: The agent's unique identifier

    Returns:
        List of key names (without the agent_id prefix)
    """
    pattern = f"{agent_id}:*"
    keys = _get_client().keys(pattern)
    prefix_len = len(f"{agent_id}:")
    return [key.decode('utf-8')[prefix_len:] for key in keys]

def get_all_working_memory(agent_id: str) -> dict[str, dict]:
    """
    Retrieve all working memory entries for an agent.

    Args:
        agent_id: The agent's unique identifier

    Returns:
        Dictionary of key -> value pairs
    """
    keys = list_working_memory_keys(agent_id)
    result = {}
    for key in keys:
        value = retrieve_working_memory(agent_id, key)
        if value is not None:
            result[key] = value
    return result

def clear_working_memory(agent_id: str) -> int:
    """
    Clear all working memory for an agent.

    Args:
        agent_id: The agent's unique identifier

    Returns:
        Number of keys deleted
    """
    pattern = f"{agent_id}:*"
    keys = _get_client().keys(pattern)
    if keys:
        return _get_client().delete(*keys)
    return 0

def update_working_memory(agent_id: str, key: str, value: dict, ttl: Optional[int] = None) -> bool:
    """
    Update an existing working memory entry, preserving TTL if not specified.

    Args:
        agent_id: The agent's unique identifier
        key: The key to update
        value: The new dictionary value
        ttl: Optional new TTL in seconds; if None, preserves existing TTL

    Returns:
        True if updated, False if key didn't exist
    """
    full_key = f"{agent_id}:{key}"

    # Check if key exists
    if not _get_client().exists(full_key):
        return False

    if ttl is None:
        # Preserve existing TTL
        remaining_ttl = _get_client().ttl(full_key)
        if remaining_ttl > 0:
            ttl = remaining_ttl
        else:
            ttl = 300  # Default if no TTL was set

    _get_client().setex(full_key, timedelta(seconds=ttl), json.dumps(value))
    return True

def extend_ttl(agent_id: str, key: str, additional_seconds: int) -> bool:
    """
    Extend the TTL of a working memory entry.

    Args:
        agent_id: The agent's unique identifier
        key: The key to extend
        additional_seconds: Seconds to add to current TTL

    Returns:
        True if extended, False if key didn't exist
    """
    full_key = f"{agent_id}:{key}"

    current_ttl = _get_client().ttl(full_key)
    if current_ttl < 0:
        return False

    new_ttl = current_ttl + additional_seconds
    return _get_client().expire(full_key, new_ttl)

def get_ttl(agent_id: str, key: str) -> int:
    """
    Get the remaining TTL for a working memory entry.

    Args:
        agent_id: The agent's unique identifier
        key: The key to check

    Returns:
        Remaining TTL in seconds, -2 if key doesn't exist, -1 if no TTL set
    """
    full_key = f"{agent_id}:{key}"
    return _get_client().ttl(full_key)

def working_memory_exists(agent_id: str, key: str) -> bool:
    """
    Check if a working memory key exists.

    Args:
        agent_id: The agent's unique identifier
        key: The key to check

    Returns:
        True if exists, False otherwise
    """
    full_key = f"{agent_id}:{key}"
    return _get_client().exists(full_key) > 0

def get_working_memory_stats(agent_id: str) -> dict:
    """
    Get statistics about an agent's working memory.

    Args:
        agent_id: The agent's unique identifier

    Returns:
        Dictionary with stats
    """
    keys = list_working_memory_keys(agent_id)
    total_size = 0

    for key in keys:
        full_key = f"{agent_id}:{key}"
        value = _get_client().get(full_key)
        if value:
            total_size += len(value)

    return {
        "key_count": len(keys),
        "total_size_bytes": total_size,
        "keys": keys
    }