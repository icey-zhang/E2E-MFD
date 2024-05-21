class BalancerRegistry:
    registry = {}


def get_method(method, *args, **kwargs):
    if method not in BalancerRegistry.registry:
        raise ValueError("Balancer named '{}' is not defined, valid methods are: {}".format(
            method, ', '.join(BalancerRegistry.registry.keys())))
    method_cls, method_args, method_kwargs = BalancerRegistry.registry[method]
    return method_cls(*method_args, *args, **method_kwargs, **kwargs)


def register(name, *args, **kwargs):
    def _register(cls):
        BalancerRegistry.registry[name] = (cls, args, kwargs)
        return cls
    return _register
