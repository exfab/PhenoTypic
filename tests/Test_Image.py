import pytest

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from phenoscope import Image
from phenoscope.data import PlateDay1

@pytest.fixture
def sample_data():
    return {
        'image':PlateDay1()
    }

def test_blank_image():
    img = Image()
    assert img.array is None
    assert img.matrix is None
    assert img.enhanced_matrix is None
    assert img.object_mask is None
    assert img.object_map is None

def test_image(sample_data):
    img = Image(sample_data['image'])

    assert img.array is not None
    assert img.matrix is not None
    assert img.enhanced_matrix is not None

    assert np.array_equal(img.object_mask , np.full(shape=img.shape, fill_value=True))
    assert np.array_equal(img.object_map, np.full(shape=img.shape, fill_value=1))

def test_image_show_array(sample_data):
    img = Image(sample_data['image'])
    fig, ax = img.show_array()
    assert fig is not None
    assert ax is not None
    plt.close(fig)

def test_image_show_matrix(sample_data):
    img = Image(sample_data['image'])
    fig, ax = img.show_matrix()
    assert fig is not None
    assert ax is not None
    plt.close(fig)

def test_image_show_enhanced_matrix(sample_data):
    img = Image(sample_data['image'])
    fig, ax = img.show_enhanced()
    assert fig is not None
    assert ax is not None
    plt.close(fig)

def test_image_show_overlay(sample_data):
    img = Image(sample_data['image'])
    fig, ax = img.show_overlay()
    assert fig is not None
    assert ax is not None
    plt.close(fig)

def test_image_show_overlay_enhanced(sample_data):
    img = Image(sample_data['image'])
    fig, ax = img.show_overlay(use_enhanced=True)
    assert fig is not None
    assert ax is not None
    plt.close(fig)

def test_image_set_metadata(sample_data):
    img = Image(sample_data['image'])
    img.name = 'test_image'
    img.set_metadata(key='test_int', value=100)
    img.set_metadata(key='test_float', value=100.0)
    img.set_metadata(key='test_bool', value=True)
    img.set_metadata(key='test_str', value='test_string')
    assert img.name == 'test_image'
    assert img.get_metadata(key='test_int') == 100
    assert img.get_metadata(key='test_float') == 100.0
    assert img.get_metadata(key='test_bool') == True
    assert img.get_metadata(key='test_str') == 'test_string'

def test_image_savez_loadz(sample_data):
    current_file_path = Path(__file__).resolve()
    test_resources_path = current_file_path.parent / 'resources'

    img = Image(sample_data['image'])
    img.name = 'test_image'
    img.set_metadata(key='test_int', value=100)
    img.set_metadata(key='test_float', value=100.0)
    img.set_metadata(key='test_bool', value=True)
    img.set_metadata(key='test_str', value='test_string')

    img.savez(test_resources_path / 'test_image_savez')