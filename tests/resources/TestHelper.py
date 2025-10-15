import importlib
import inspect
import pkgutil
import time
import functools


def timeit(func):
    """Decorator to measure the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # High-resolution timer
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds")
        print(f"executed in ({elapsed_time:.6f} seconds)")
        return result

    return wrapper


def walk_package_for_class(pkg, cls):
    """Yield (qualified_name, obj) for every public, top‑level object in *pkg*
    and all of its sub‑modules, skipping module objects themselves. this collects all image operations for testing."""
    modules = [pkg]  # start with the root
    if hasattr(pkg, "__path__"):  # add all sub‑modules
        modules += [
            importlib.import_module(name)
            for _, name, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".") \
            if not name.split(".")[-1].startswith("_")  # Skip modules with names starting with underscore

        ]

    seen = set()
    for mod in modules:
        if mod.__name__.startswith("_"):
            continue
        for attr in dir(mod):
            if attr.startswith("_"):
                continue

            obj = getattr(mod, attr)
            if (inspect.ismodule(obj)
                    or inspect.isabstract(obj)
                    or not isinstance(obj, type)
                    or not issubclass(obj, cls)):
                continue

            qualname = f"{mod.__name__}.{attr}"
            if qualname not in seen:
                seen.add(qualname)
                yield qualname, obj
