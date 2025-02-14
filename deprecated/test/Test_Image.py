import pytest

import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from phenoscope import Image
from phenoscope.data import load_plate_12hr

@pytest.fixture
def sample_data():
    return {
        'image':load_plate_12hr()
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

    assert np.array_equal(img.object_mask , np.full(shape=img.shape, fill_value=1))
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
    test_filepath = test_resources_path / 'test_image_savez.psnpz'

    img = Image(sample_data['image'])

    original_array = img.array
    original_matrix = img.matrix
    original_enhanced_matrix = img.enhanced_matrix
    original_object_mask = img.object_mask
    original_object_map = img.object_map

    img.name = 'test_image'
    img.set_metadata(key='test_int', value=100)
    img.set_metadata(key='test_float', value=100.0)
    img.set_metadata(key='test_bool', value=True)
    img.set_metadata(key='test_str', value='test_string')

    if test_filepath.exists(): os.remove(test_filepath)
    assert test_filepath.exists() is False

    img.savez(test_filepath)

    assert test_filepath.exists()
    new_img = Image().loadz(test_filepath)
    assert new_img is not None
    assert np.array_equal(original_array, new_img.array)
    assert np.array_equal(original_matrix, new_img.matrix)
    assert np.array_equal(original_enhanced_matrix, new_img.det_matrix)
    assert np.array_equal(original_object_mask, new_img.object_mask)
    assert np.array_equal(original_object_map, new_img.object_map)

    assert new_img.name == 'test_image'
    assert new_img.get_metadata(key='test_int') == 100
    assert new_img.get_metadata(key='test_float') == 100.0
    assert new_img.get_metadata(key='test_bool') == True
    assert new_img.get_metadata(key='test_str') == 'test_string'
    assert img._metadata==new_img._metadata




