# dynamic_imports.py
import importlib
import pkgutil
import inspect
import os

def load_classes_and_methods(submodule_folder):
    __all__ = []

    # Iterate through all modules in the submodule directory
    for module_info in pkgutil.iter_modules([submodule_folder]):
        module_name = module_info.name
        # Dynamically import the module
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Add classes or methods dynamically to the globals()
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                globals()[name] = obj
                __all__.append(name)
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module.__name__:
                globals()[name] = obj
                __all__.append(name)

    return __all__

