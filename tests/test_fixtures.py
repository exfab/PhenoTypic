import importlib
import inspect
import pkgutil

import numpy as np
import pytest

import phenotypic


def param2array(tag):
    from phenotypic.data import load_early_colony, load_colony, load_plate_12hr, load_plate_72hr

    match tag:
        case 'km-plate-12hr':
            return load_plate_12hr()
        case 'km-plate-72hr':
            return load_plate_72hr()
        case 'km-colony-12hr':
            return load_early_colony()
        case 'km-colony-72hr':
            return load_colony()
        case 'black-square':
            return np.full(shape=(100, 100), fill_value=0.0)
        case 'white-square':
            return np.full(shape=(100, 100), fill_value=1.0)
        case _:
            raise ValueError(f'Invalid tag: {tag}')


def param2array_plus_imformat(tag):
    from phenotypic.data import load_early_colony, load_colony, load_plate_12hr, load_plate_72hr

    match tag:
        case 'km-plate-12hr':
            return load_plate_12hr(), None, 'RGB'
        case 'km-plate-72hr':
            return load_plate_72hr(), 'RGB', 'RGB'
        case 'km-colony-12hr':
            return load_early_colony(), 'RGB', 'RGB'
        case 'km-colony-72hr':
            return load_colony(), 'RGB', 'RGB'
        case 'black-square':
            return np.full(shape=(100, 100), fill_value=0.0), None, 'Grayscale'
        case 'white-square':
            return np.full(shape=(100, 100), fill_value=1.0), 'Grayscale', 'Grayscale'
        case _:
            raise ValueError(f'Invalid tag: {tag}')


@pytest.fixture(
    scope='session',
    params=[
        pytest.param('km-plate-12hr', id='Plate-None-RGB', ),
        pytest.param('km-plate-72hr', id='Plate-RGB-RGB', ),
        pytest.param('km-colony-12hr', id='Colony-RGB-RGB', ),
        pytest.param('km-colony-72hr', id='Colony-RGB-RGB', ),
        pytest.param('black-square', id='Black-Square-Grayscale', ),
        pytest.param('white-square', id='White-Square-Grayscale', )
    ]
)
def sample_image_array_with_imformat(request):
    """Fixture that returns (image_array, input_imformat, true_imformat)"""
    arr, inp_fmt, true_fmt = param2array_plus_imformat(request.param)
    return arr, inp_fmt, true_fmt


@pytest.fixture(
    scope='session',
    params=[
        pytest.param('km-plate-12hr', id='Plate-None-RGB', ),
        pytest.param('km-plate-72hr', id='Plate-RGB-RGB', ),
        pytest.param('km-colony-12hr', id='Colony-RGB-RGB', ),
        pytest.param('km-colony-72hr', id='Colony-RGB-RGB', ),
        pytest.param('black-square', id='Black-Square-Grayscale', ),
        pytest.param('white-square', id='White-Square-Grayscale', )
    ]
)
def sample_image_array(request):
    """Fixture that returns (image_array, input_imformat, true_imformat)"""
    arr = param2array(request.param)
    return arr


@pytest.fixture(
    scope='session',
    params=[
        pytest.param('km-plate-12hr', id='km-plate-12hr-GridImage', ),
        pytest.param('km-plate-72hr', id='km-plate-72hr-GridImage', )
    ]
)
def plate_grid_images(request):
    import phenotypic
    array = param2array(request.param)
    return phenotypic.GridImage(array)


@pytest.fixture(scope='session',
                params=[
                    pytest.param('km-plate-12hr', id='km-plate-12hr-GridImage-detected', ),
                    pytest.param('km-plate-72hr', id='km-plate-72hr-GridImage-detected', )
                ]
                )
def plate_grid_images_with_detection(request):
    import phenotypic
    image = phenotypic.GridImage(param2array(request.param))
    return phenotypic.detection.OtsuDetector().apply(image)


def walk_package(pkg):
    """Yield (qualified_name, obj) for every public, top‑level object in *pkg*
    and all of its sub‑modules, skipping module objects themselves."""
    modules = [pkg]  # start with the root
    if hasattr(pkg, "__path__"):  # add all sub‑modules
        modules += [
            importlib.import_module(name)
            for _, name, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")\
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

            qualname = f"{mod.__name__}.{attr}"
            if qualname not in seen:
                seen.add(qualname)
                yield qualname, obj


_public = list(walk_package(phenotypic))


def walk_package_for_operations(pkg):
    """Yield (qualified_name, obj) for every public, top‑level object in *pkg*
    and all of its sub‑modules, skipping module objects themselves. this collects all image operations for testing."""
    modules = [pkg]  # start with the root
    if hasattr(pkg, "__path__"):  # add all sub‑modules
        modules += [
            importlib.import_module(name)
            for _, name, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")\
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

            if not isinstance(obj, type):   # make sure object is a class object
                continue

            if not issubclass(obj, phenotypic.abstract.ImageOperation):
                continue

            qualname = f"{mod.__name__}.{attr}"
            if qualname not in seen:
                seen.add(qualname)
                yield qualname, obj


_image_operations = list(walk_package_for_operations(phenotypic))


def walk_package_for_measurements(pkg):
    """Yield (qualified_name, obj) for every public, top‑level object in *pkg*
    and all of its sub‑modules, skipping module objects themselves. this collects all image measurement modules for testing."""
    modules = [pkg]  # start with the root
    if hasattr(pkg, "__path__"):  # add all sub‑modules
        modules += [
            importlib.import_module(name)
            for _, name, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")\
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

            if not isinstance(obj, type): # make sure object is a class object
                continue

            if not issubclass(obj, phenotypic.abstract.MeasureFeatures):
                continue

            qualname = f"{mod.__name__}.{attr}"
            if qualname not in seen:
                seen.add(qualname)
                yield qualname, obj


_image_measurements = list(walk_package_for_measurements(phenotypic))
