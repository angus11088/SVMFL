# registry.py

class Registry:
    def __init__(self, name):
        self._name = name
        self._dict = {}

    def register(self):
        def decorator(fn):
            self._dict[fn.__name__] = fn
            return fn
        return decorator

    def get(self, name):
        if name not in self._dict:
            raise ValueError(f"[{self._name}] Cannot find: {name}")
        return self._dict[name]

    def __str__(self):
        return f"[Registry-{self._name}] contains: {list(self._dict.keys())}"
