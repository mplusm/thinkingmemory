import redis
import json
from datetime import timedelta

redis_client = redis.Redis(host="localhost", port=6379, db=0)

def store_working_memory(agent_id: str, key: str, value: dict, ttl: int = 300):
    redis_client.setex(f"{agent_id}:{key}", timedelta(seconds=ttl), json.dumps(value))

def retrieve_working_memory(agent_id: str, key: str):
    value = redis_client.get(f"{agent_id}:{key}")
    return json.loads(value) if value else None