from ninflam.core.exceptions import RegistryError

_REGISTRY = {}

def register(kind: str, name: str):
    def deco(cls):
        _REGISTRY[(kind, name)] = cls
        return cls
    return deco

def get(kind: str, name: str):
    key = (kind, name)
    if key not in _REGISTRY:
        raise RegistryError(f"Nothing registered for kind={kind!r}, name={name!r}")
    return _REGISTRY[key]
