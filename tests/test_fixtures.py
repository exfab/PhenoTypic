import numpy as np
import pytest


def param2array(tag):
    from phenotypic.data import load_colony_12_hr, load_colony_72hr, load_plate_12hr, load_plate_72hr

    match tag:
        case 'km-plate-12hr':
            return load_plate_12hr()
        case 'km-plate-72hr':
            return load_plate_72hr()
        case 'km-colony-12hr':
            return load_colony_12_hr()
        case 'km-colony-72hr':
            return load_colony_72hr()
        case 'black-square':
            return np.full(shape=(100, 100), fill_value=0)
        case 'white-square':
            return np.full(shape=(100, 100), fill_value=1)
        case _:
            raise ValueError(f'Invalid tag: {tag}')


def param2array_plus_imformat(tag):
    from phenotypic.data import load_colony_12_hr, load_colony_72hr, load_plate_12hr, load_plate_72hr

    match tag:
        case 'km-plate-12hr':
            return load_plate_12hr(), None, 'RGB'
        case 'km-plate-72hr':
            return load_plate_72hr(), 'RGB', 'RGB'
        case 'km-colony-12hr':
            return load_colony_12_hr(), 'RGB', 'RGB'
        case 'km-colony-72hr':
            return load_colony_72hr(), 'RGB', 'RGB'
        case 'black-square':
            return np.full(shape=(100, 100), fill_value=0), None, 'Grayscale'
        case 'white-square':
            return np.full(shape=(100, 100), fill_value=1), 'Grayscale', 'Grayscale'
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
