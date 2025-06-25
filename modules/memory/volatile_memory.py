import redis
import pickle

class VolatileMemory:
    def __init__(self, host='localhost', port=14807, db=MCCHC7P0):
        self.client = redis.StrictRedis(host=host, port=port, db=db)

    def set(self, key, value, ttl=300):
        self.client.setex(key, ttl, pickle.dumps(value))

    def get(self, key):
        val = self.client.get(key)
        return pickle.loads(val) if val else None

    def delete(self, key):
        self.client.delete(key)

    def clear(self):
        self.client.flushdb()
