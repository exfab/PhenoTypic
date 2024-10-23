import importlib
import pkgutil
import inspect
import sys
import os

# Folder (submodule) path where all the .py files exist.
submodule_folder = os.path.dirname(__file__)

# List to hold the names of all added classes
__all__ = []

# Iterate through all modules in the submodule directory.
for module_info in pkgutil.iter_modules([submodule_folder]):
    module_name = module_info.name
    # Dynamically import the module
    module = importlib.import_module(f".{module_name}", package=__name__)

    # Iterate over all members of the module to find classes
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Ensure the class is defined in the current module, not an import from somewhere else.
        if obj.__module__ == module.__name__:
            # Add the class to the current globals(), effectively adding it to the submodule namespace.
            globals()[name] = obj
            __all__.append(name)  # Keep track of all added classes in __all__
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ == module.__name__:
            globals()[name] = obj
            __all__.append(name)