from operator import attrgetter

def attrsetter(obj, attr, val):
    prefix, name = attr.rsplit('.', 1)
    getter = attrgetter(prefix)
    parent = getter(obj)
    return setattr(parent, name, val)
