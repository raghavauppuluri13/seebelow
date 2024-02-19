def dict_from_class(cls):
    excluded_keys = set(dir(type("dummy", (object,), {})))
    return {
        key: value for key, value in cls.__dict__.items() if key not in excluded_keys
    }
