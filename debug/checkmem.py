from memory_profiler import profile

globals()['profile'] = profile

import types
import inspect
import pkgutil
import importlib

import psutil
import os
import time
from functools import wraps


def profile_ram(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        before = process.memory_info().rss  # in bytes
        t0 = time.perf_counter()

        result = func(*args, **kwargs)

        after = process.memory_info().rss
        t1 = time.perf_counter()
        delta = (after - before)/1024 ** 2  # in MB

        print(f"[RAM] {func.__qualname__} used {delta:.3f} MB in {t1 - t0:.3f}s")
        return result

    return wrapper


def auto_profile_module(module):
    for name in dir(module):
        obj = getattr(module, name)

        # Decorate top-level functions
        if isinstance(obj, types.FunctionType):
            setattr(module, name, profile(obj))

        # Decorate class methods
        elif isinstance(obj, type):
            for attr_name, attr in vars(obj).items():
                if isinstance(attr, (types.FunctionType, types.MethodType)):
                    setattr(obj, attr_name, profile(attr))


# New function: profile all submodules of a package
def auto_profile_package(pkg):
    # Profile the root package module
    auto_profile_module(pkg)
    # Recursively import and profile all submodules
    if hasattr(pkg, "__path__"):
        for _, module_name, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            module = importlib.import_module(module_name)
            auto_profile_module(module)


import types


def auto_profile_module_ram(module):
    for name in dir(module):
        obj = getattr(module, name)

        # Wrap functions directly in the module
        if isinstance(obj, types.FunctionType):
            setattr(module, name, profile_ram(obj))

        # Wrap methods in classes within the module
        elif isinstance(obj, type):
            for attr_name, attr in vars(obj).items():
                if isinstance(attr, (types.FunctionType, types.MethodType)):
                    setattr(obj, attr_name, profile_ram(attr))


import phenotypic

auto_profile_package(phenotypic)


def walk_package_for_measurements(pkg):
    """Yield (qualified_name, obj) for every public, top‑level object in *pkg*
    and all of its sub‑modules, skipping module objects themselves. this collects all image measurement modules for testing."""
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
            if inspect.ismodule(obj):
                continue

            if not isinstance(obj, type):  # make sure object is a class object
                continue

            if not issubclass(obj, phenotypic.abstract.MeasureFeatures):
                continue

            qualname = f"{mod.__name__}.{attr}"
            if qualname not in seen:
                seen.add(qualname)
                yield qualname, obj


import phenotypic
from phenotypic.data import load_plate_12hr
from phenotypic.detect import WatershedDetector
import pandas as pd


def test_measurement(qualname, obj):
    """The goal of this test is to ensure that all operations are callable with basic functionality,
     and return a valid Image object."""
    try:
        print(f"Testing {qualname}")
        image = phenotypic.GridImage(load_plate_12hr())
        image = WatershedDetector().apply(image)
        assert isinstance(obj().measure(image), pd.DataFrame)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print(f"Failed on {qualname} - e")


for qualname, obj in walk_package_for_measurements(phenotypic):
    test_measurement(qualname, obj)


def walk_package_for_operations(pkg):
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
            if inspect.ismodule(obj):
                continue

            if not isinstance(obj, type):  # make sure object is a class object
                continue

            if not issubclass(obj, phenotypic.abstract.ImageOperation):
                continue

            qualname = f"{mod.__name__}.{attr}"
            if qualname not in seen:
                seen.add(qualname)
                yield qualname, obj


def test_operation(qualname, obj):
    """The goal of this test is to ensure that all operations are callable with basic functionality
     and return a valid Image object."""
    try:
        print(f"Testing {qualname}")
        image = phenotypic.GridImage(load_plate_12hr())
        image = WatershedDetector().apply(image)
        assert obj().apply(image).isempty() is False
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print(f"Failed on {qualname} - e")


for qualname, obj in walk_package_for_operations(phenotypic):
    test_operation(qualname, obj)
