import memcache

class VolatileMemory:
    def __init__(self, server='127.0.0.1:11211'):
        self.client = memcache.Client([server])

    def get(self, key):
        return self.client.get(key)

    def set(self, key, value, ttl=300):
        return self.client.set(key, value, time=ttl)

    def delete(self, key):
        return self.client.delete(key)

    def clear(self):
        return self.client.flush_all()
