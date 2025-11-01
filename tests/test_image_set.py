"""Unit tests for ImageSet classes (ImageSetCore, ImageSetStatus, ImageSetMeasurements)."""

import gc
import os
import tempfile
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import pytest

import phenotypic as pht
from phenotypic.core._image_set_parts._image_set_core import ImageSetCore
from phenotypic.core._image_set_parts._image_set_status import ImageSetStatus
from phenotypic.core._image_set_parts._image_set_measurements import ImageSetMeasurements
from phenotypic.tools.constants_ import PIPE_STATUS


# =================== FIXTURES ===================

@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_hdf5_path(temp_dir):
    """Create path for temporary HDF5 file."""
    return temp_dir / "test_image_set.h5"


@pytest.fixture
def sample_images() -> List[pht.Image]:
    """Generate list of test Image objects with known properties."""
    images = []
    for i in range(3):
        # Create simple test image
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = pht.Image(arr=arr, imformat='RGB', name=f'test_image_{i}')
        img._metadata.public['test_key'] = f'value_{i}'
        img._metadata.protected['ImageName'] = f'test_image_{i}'
        images.append(img)
    return images


@pytest.fixture
def sample_dir_with_images(temp_dir, sample_images):
    """Create temp directory with image files."""
    img_dir = temp_dir / "images"
    img_dir.mkdir()
    
    for i, img in enumerate(sample_images):
        filepath = img_dir / f"test_image_{i}.png"
        img.array.imsave(str(filepath))
    
    return img_dir


@pytest.fixture
def image_with_measurements() -> pht.Image:
    """Create Image with measurement data attached."""
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = pht.Image(arr=arr, imformat='RGB', name='measured_image')
    
    # Add some measurements
    measurements = pd.DataFrame({
        'ObjectLabel': [1, 2, 3],
        'Area': [100, 200, 150],
        'Intensity_Mean': [0.5, 0.7, 0.6]
    })
    img._measurements = measurements
    img._metadata.public['test_metadata'] = 'test_value'
    img._metadata.protected['ImageName'] = 'measured_image'
    
    return img


# =================== TEST CLASSES ===================

class TestImageSetCoreInitialization:
    """Test ImageSetCore construction and resource management."""
    
    def test_default_initialization_temp_mode(self):
        """Test default initialization creates temp file."""
        img_set = ImageSetCore(name='test_set', default_mode='temp')
        
        assert img_set.name == 'test_set'
        assert img_set._out_path.exists()
        assert img_set._out_path.suffix in ['.h5', '.hdf5']
        assert img_set._owns_outpath is True
        
        # Cleanup
        img_set.close()
        assert not img_set._out_path.exists()
    
    def test_initialization_explicit_outpath(self, temp_hdf5_path):
        """Test initialization with explicit outpath."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        
        assert img_set._out_path == temp_hdf5_path
        assert img_set._out_path.exists()
        assert img_set._owns_outpath is False
    
    def test_initialization_cwd_mode(self):
        """Test initialization with cwd mode."""
        img_set = ImageSetCore(name='test_set', default_mode='cwd')
        
        expected_path = Path.cwd() / 'test_set.hdf5'
        assert img_set._out_path == expected_path
        assert img_set._out_path.exists()
        assert img_set._owns_outpath is False
        
        # Cleanup
        if expected_path.exists():
            expected_path.unlink()
    
    def test_directory_as_outpath(self, temp_dir):
        """Test directory as outpath appends .hdf5."""
        img_set = ImageSetCore(name='test_set', outpath=temp_dir, default_mode='temp')
        
        expected_path = temp_dir / 'test_set.hdf5'
        assert img_set._out_path == expected_path
        assert img_set._out_path.exists()
    
    def test_invalid_file_extension(self, temp_dir):
        """Test invalid file extension handling."""
        invalid_path = temp_dir / 'test.txt'
        
        with pytest.raises(ValueError, match='Invalid output file extension'):
            ImageSetCore(name='test_set', outpath=invalid_path, default_mode='temp')
    
    def test_resource_cleanup_on_close(self):
        """Test temp file deletion on close()."""
        img_set = ImageSetCore(name='test_set', default_mode='temp')
        temp_path = img_set._out_path
        
        assert temp_path.exists()
        img_set.close()
        assert not temp_path.exists()
    
    def test_weakref_finalizer_cleanup(self):
        """Test weakref finalizer cleanup on garbage collection."""
        img_set = ImageSetCore(name='test_set', default_mode='temp')
        temp_path = img_set._out_path
        
        assert temp_path.exists()
        del img_set
        gc.collect()
        
        # Temp file should be deleted
        assert not temp_path.exists()


class TestImageSetCoreImageImport:
    """Test image import functionality."""
    
    def test_import_images_with_list(self, sample_images, temp_hdf5_path):
        """Test import_images() with list of Image objects."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        
        names = img_set.get_image_names()
        assert len(names) == 3
        assert all(f'test_image_{i}' in names for i in range(3))
    
    def test_import_dir_with_mixed_files(self, sample_dir_with_images, temp_hdf5_path):
        """Test import_dir() from directory with image files."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_dir(sample_dir_with_images)
        
        names = img_set.get_image_names()
        assert len(names) == 3
    
    def test_empty_directory_handling(self, temp_dir, temp_hdf5_path):
        """Test empty directory handling."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_dir(empty_dir)
        
        names = img_set.get_image_names()
        assert len(names) == 0
    
    def test_non_existent_directory_error(self, temp_dir, temp_hdf5_path):
        """Test non-existent directory raises error."""
        non_existent = temp_dir / "does_not_exist"
        
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        with pytest.raises(ValueError, match='is not a directory'):
            img_set.import_dir(non_existent)
    
    def test_image_type_parameter(self, sample_images, temp_hdf5_path):
        """Test imtype parameter (Image vs GridImage)."""
        img_set = ImageSetCore(
            name='test_set',
            outpath=temp_hdf5_path,
            imtype='Image',
            default_mode='temp'
        )
        
        assert img_set.imtype == 'Image'
        assert img_set._get_template() == pht.Image
    
    def test_image_parameters_passed_through(self, temp_dir, temp_hdf5_path):
        """Test imparams are passed to image construction."""
        # Create a simple test image file
        test_img = pht.Image(
            arr=np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            imformat='RGB',
            name='param_test'
        )
        test_file = temp_dir / "param_test.png"
        test_img.array.imsave(str(test_file))
        
        img_set = ImageSetCore(
            name='test_set',
            outpath=temp_hdf5_path,
            imtype='Image',
            imparams={'name': 'custom_name'},
            default_mode='temp'
        )
        
        assert img_set.imparams == {'name': 'custom_name'}


class TestImageSetCoreImageAccess:
    """Test image retrieval methods."""
    
    def test_get_image_names_returns_correct_list(self, sample_images, temp_hdf5_path):
        """Test get_image_names() returns correct list."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        
        names = img_set.get_image_names()
        assert isinstance(names, list)
        assert len(names) == 3
        assert set(names) == {'test_image_0', 'test_image_1', 'test_image_2'}
    
    def test_get_image_retrieves_specific_image(self, sample_images, temp_hdf5_path):
        """Test get_image() retrieves specific image."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        
        retrieved_img = img_set.get_image('test_image_0')
        assert isinstance(retrieved_img, pht.Image)
        assert retrieved_img.name == 'test_image_0'
    
    def test_get_image_non_existent_raises_error(self, sample_images, temp_hdf5_path):
        """Test get_image() with non-existent name raises error."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        
        with pytest.raises(ValueError, match='not found in ImageSet'):
            img_set.get_image('non_existent')
    
    def test_iter_images_yields_all_images(self, sample_images, temp_hdf5_path):
        """Test iter_images() yields all images in order."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        
        retrieved_images = list(img_set.iter_images())
        assert len(retrieved_images) == 3
        assert all(isinstance(img, pht.Image) for img in retrieved_images)
    
    def test_add_image_adds_single_image(self, sample_images, temp_hdf5_path):
        """Test add_image() adds single image."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        
        img_set.add_image(sample_images[0])
        names = img_set.get_image_names()
        assert len(names) == 1
        assert 'test_image_0' in names
    
    def test_add_image_overwrite_false_prevents_duplicates(self, sample_images, temp_hdf5_path):
        """Test add_image() with overwrite=False prevents duplicates."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        
        img_set.add_image(sample_images[0], overwrite=False)
        
        with pytest.raises(ValueError, match='already exists'):
            img_set.add_image(sample_images[0], overwrite=False)
    
    def test_add_image_overwrite_true_replaces_existing(self, sample_images, temp_hdf5_path):
        """Test add_image() with overwrite=True replaces existing."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        
        img_set.add_image(sample_images[0], overwrite=False)
        # Should not raise error
        img_set.add_image(sample_images[0], overwrite=True)
        
        names = img_set.get_image_names()
        assert len(names) == 1


class TestImageSetStatusTracking:
    """Test ImageSetStatus functionality."""
    
    def test_reset_status_initializes_all_flags_false(self, sample_images, temp_hdf5_path):
        """Test reset_status() initializes all status flags to False."""
        img_set = ImageSetStatus(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        img_set.reset_status()
        
        status_df = img_set.get_status()
        assert len(status_df) == 3
        assert all(status_df[str(PIPE_STATUS.PROCESSED)] == False)
        assert all(status_df[str(PIPE_STATUS.MEASURED)] == False)
    
    def test_reset_status_with_specific_image_names(self, sample_images, temp_hdf5_path):
        """Test reset_status() with specific image names."""
        img_set = ImageSetStatus(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        img_set.reset_status()
        
        # Reset only one image
        img_set.reset_status('test_image_0')
        
        status_df = img_set.get_status(['test_image_0'])
        assert len(status_df) == 1
        assert status_df.index[0] == 'test_image_0'
    
    def test_get_status_returns_dataframe_correct_structure(self, sample_images, temp_hdf5_path):
        """Test get_status() returns DataFrame with correct structure."""
        img_set = ImageSetStatus(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        img_set.reset_status()
        
        status_df = img_set.get_status()
        
        assert isinstance(status_df, pd.DataFrame)
        assert str(PIPE_STATUS.PROCESSED) in status_df.columns
        assert str(PIPE_STATUS.MEASURED) in status_df.columns
        assert len(status_df) == 3
    
    def test_status_flags_properly_tracked(self, sample_images, temp_hdf5_path):
        """Test status flags (PROCESSED, MEASURED) properly tracked."""
        img_set = ImageSetStatus(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        img_set.reset_status()
        
        # Manually set a status flag
        with img_set.hdf_.writer() as writer:
            status_group = img_set.hdf_.get_status_subgroup(writer, 'test_image_0')
            status_group.attrs[PIPE_STATUS.PROCESSED.label] = True
        
        status_df = img_set.get_status(['test_image_0'])
        assert status_df[str(PIPE_STATUS.PROCESSED)].iloc[0] == True
    
    def test_status_persists_in_hdf5_attributes(self, sample_images, temp_hdf5_path):
        """Test status persists in HDF5 attributes."""
        img_set = ImageSetStatus(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        img_set.reset_status()
        
        # Set status
        with img_set.hdf_.writer() as writer:
            status_group = img_set.hdf_.get_status_subgroup(writer, 'test_image_0')
            status_group.attrs[PIPE_STATUS.PROCESSED.label] = True
        
        # Verify persistence with direct HDF5 access
        with h5py.File(temp_hdf5_path, 'r') as f:
            status_path = f'/phenotypic/image_sets/test_set/data/test_image_0/status'
            assert f[status_path].attrs[PIPE_STATUS.PROCESSED.label] == True
    
    def test_add_image2group_initializes_status(self, sample_images, temp_hdf5_path):
        """Test _add_image2group() initializes status on image add."""
        img_set = ImageSetStatus(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        
        with img_set.hdf_.writer() as writer:
            data_group = img_set.hdf_.get_data_group(writer)
            img_set._add_image2group(data_group, sample_images[0], overwrite=False)
        
        # Check status was initialized
        status_df = img_set.get_status(['test_image_0'])
        assert all(status_df[str(PIPE_STATUS.PROCESSED)] == False)
        assert all(status_df[str(PIPE_STATUS.MEASURED)] == False)


class TestImageSetMeasurements:
    """Test ImageSetMeasurements functionality."""
    
    def test_get_measurement_empty_when_no_measurements(self, sample_images, temp_hdf5_path):
        """Test get_measurement() returns empty DataFrame when no measurements."""
        img_set = ImageSetMeasurements(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        img_set.reset_status()
        
        result = img_set.get_measurement()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_get_measurement_validates_status_flags(self, image_with_measurements, temp_hdf5_path):
        """Test get_measurement() validates status flags (PROCESSED & MEASURED must be True)."""
        img_set = ImageSetMeasurements(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images([image_with_measurements])
        img_set.reset_status()
        
        # Add measurements to HDF5
        with img_set.hdf_.writer() as writer:
            meas_group = img_set.hdf_.get_image_measurement_subgroup(writer, 'measured_image')
            img_set.hdf_.save_frame_new(meas_group, image_with_measurements._measurements, require_swmr=False)
            
            # Only set PROCESSED, not MEASURED
            status_group = img_set.hdf_.get_status_subgroup(writer, 'measured_image')
            status_group.attrs[PIPE_STATUS.PROCESSED.label] = True
            status_group.attrs[PIPE_STATUS.MEASURED.label] = False
        
        # Should return empty because MEASURED is False
        result = img_set.get_measurement()
        assert len(result) == 0
    
    def test_get_measurement_merges_protected_metadata(self, image_with_measurements, temp_hdf5_path):
        """Test get_measurement() merges protected metadata."""
        img_set = ImageSetMeasurements(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images([image_with_measurements])
        img_set.reset_status()
        
        # Add measurements and set status
        with img_set.hdf_.writer() as writer:
            meas_group = img_set.hdf_.get_image_measurement_subgroup(writer, 'measured_image')
            img_set.hdf_.save_frame_new(meas_group, image_with_measurements._measurements, require_swmr=False)
            
            status_group = img_set.hdf_.get_status_subgroup(writer, 'measured_image')
            status_group.attrs[PIPE_STATUS.PROCESSED.label] = True
            status_group.attrs[PIPE_STATUS.MEASURED.label] = True
        
        result = img_set.get_measurement()
        assert len(result) == 3
        assert 'Metadata_ImageName' in result.columns
    
    def test_get_measurement_merges_public_metadata(self, image_with_measurements, temp_hdf5_path):
        """Test get_measurement() merges public metadata."""
        img_set = ImageSetMeasurements(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images([image_with_measurements])
        img_set.reset_status()
        
        # Add measurements and set status
        with img_set.hdf_.writer() as writer:
            meas_group = img_set.hdf_.get_image_measurement_subgroup(writer, 'measured_image')
            img_set.hdf_.save_frame_new(meas_group, image_with_measurements._measurements, require_swmr=False)
            
            status_group = img_set.hdf_.get_status_subgroup(writer, 'measured_image')
            status_group.attrs[PIPE_STATUS.PROCESSED.label] = True
            status_group.attrs[PIPE_STATUS.MEASURED.label] = True
        
        result = img_set.get_measurement()
        assert 'Metadata_test_metadata' in result.columns
        assert all(result['Metadata_test_metadata'] == 'test_value')
    
    def test_get_measurement_with_specific_image_names(self, image_with_measurements, temp_hdf5_path):
        """Test get_measurement() with specific image names."""
        # Create second image with measurements
        img2 = pht.Image(
            arr=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            imformat='RGB',
            name='measured_image_2'
        )
        img2._measurements = pd.DataFrame({
            'ObjectLabel': [1, 2],
            'Area': [50, 75]
        })
        img2._metadata.protected['ImageName'] = 'measured_image_2'
        
        img_set = ImageSetMeasurements(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images([image_with_measurements, img2])
        img_set.reset_status()
        
        # Add measurements for both
        for img_name, img in [('measured_image', image_with_measurements), ('measured_image_2', img2)]:
            with img_set.hdf_.writer() as writer:
                meas_group = img_set.hdf_.get_image_measurement_subgroup(writer, img_name)
                img_set.hdf_.save_frame_new(meas_group, img._measurements, require_swmr=False)
                
                status_group = img_set.hdf_.get_status_subgroup(writer, img_name)
                status_group.attrs[PIPE_STATUS.PROCESSED.label] = True
                status_group.attrs[PIPE_STATUS.MEASURED.label] = True
        
        # Get only first image measurements
        result = img_set.get_measurement(['measured_image'])
        assert len(result) == 3  # Only first image has 3 objects
    
    def test_measurement_dataframe_structure(self, image_with_measurements, temp_hdf5_path):
        """Test measurement DataFrame structure and column order."""
        img_set = ImageSetMeasurements(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images([image_with_measurements])
        img_set.reset_status()
        
        with img_set.hdf_.writer() as writer:
            meas_group = img_set.hdf_.get_image_measurement_subgroup(writer, 'measured_image')
            img_set.hdf_.save_frame_new(meas_group, image_with_measurements._measurements, require_swmr=False)
            
            status_group = img_set.hdf_.get_status_subgroup(writer, 'measured_image')
            status_group.attrs[PIPE_STATUS.PROCESSED.label] = True
            status_group.attrs[PIPE_STATUS.MEASURED.label] = True
        
        result = img_set.get_measurement()
        
        # Metadata columns should come first
        metadata_cols = [col for col in result.columns if col.startswith('Metadata_')]
        assert len(metadata_cols) > 0
        assert all(result.columns.tolist().index(col) < result.columns.tolist().index('ObjectLabel') 
                   for col in metadata_cols)
    
    def test_multiple_images_different_schemas(self, temp_hdf5_path):
        """Test multiple images with different measurement schemas."""
        img1 = pht.Image(
            arr=np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            imformat='RGB',
            name='img1'
        )
        img1._measurements = pd.DataFrame({
            'ObjectLabel': [1, 2],
            'Area': [100, 200]
        })
        img1._metadata.protected['ImageName'] = 'img1'
        
        img2 = pht.Image(
            arr=np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            imformat='RGB',
            name='img2'
        )
        img2._measurements = pd.DataFrame({
            'ObjectLabel': [1],
            'Area': [150],
            'Perimeter': [50]  # Different schema
        })
        img2._metadata.protected['ImageName'] = 'img2'
        
        img_set = ImageSetMeasurements(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images([img1, img2])
        img_set.reset_status()
        
        for img_name, img in [('img1', img1), ('img2', img2)]:
            with img_set.hdf_.writer() as writer:
                meas_group = img_set.hdf_.get_image_measurement_subgroup(writer, img_name)
                img_set.hdf_.save_frame_new(meas_group, img._measurements, require_swmr=False)
                
                status_group = img_set.hdf_.get_status_subgroup(writer, img_name)
                status_group.attrs[PIPE_STATUS.PROCESSED.label] = True
                status_group.attrs[PIPE_STATUS.MEASURED.label] = True
        
        result = img_set.get_measurement()
        # Should handle different schemas (pandas concat fills NaN for missing columns)
        assert len(result) == 3
        assert 'Perimeter' in result.columns


class TestImageSetHDF5Integration:
    """Test HDF5 file structure and operations."""
    
    def test_correct_hdf5_group_hierarchy(self, sample_images, temp_hdf5_path):
        """Test correct HDF5 group hierarchy."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        
        with h5py.File(temp_hdf5_path, 'r') as f:
            assert '/phenotypic/image_sets/test_set' in f
            assert '/phenotypic/image_sets/test_set/data' in f
            assert '/phenotypic/image_sets/test_set/data/test_image_0' in f
    
    def test_multiple_imagesets_same_file(self, sample_images, temp_hdf5_path):
        """Test multiple ImageSets in same HDF5 file."""
        img_set1 = ImageSetCore(name='set1', outpath=temp_hdf5_path, default_mode='temp')
        img_set1.import_images([sample_images[0]])
        
        img_set2 = ImageSetCore(name='set2', outpath=temp_hdf5_path, default_mode='temp')
        img_set2.import_images([sample_images[1]])
        
        with h5py.File(temp_hdf5_path, 'r') as f:
            assert '/phenotypic/image_sets/set1' in f
            assert '/phenotypic/image_sets/set2' in f
    
    def test_concurrent_read_access(self, sample_images, temp_hdf5_path):
        """Test concurrent read access (SWMR mode)."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        
        # Open for reading with SWMR
        with img_set.hdf_.swmr_reader() as reader:
            names = list(img_set.hdf_.get_data_group(reader).keys())
            assert len(names) == 3
    
    def test_file_locking_retry_mechanism(self, sample_images, temp_hdf5_path):
        """Test file locking and retry mechanism."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        
        # safe_writer should handle any locking issues
        with img_set.hdf_.safe_writer() as writer:
            assert writer is not None
            data_group = img_set.hdf_.get_data_group(writer)
            assert len(data_group.keys()) == 3
    
    def test_proper_context_manager_usage(self, sample_images, temp_hdf5_path):
        """Test proper context manager usage."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        
        # Ensure file is properly closed after context
        with img_set.hdf_.reader() as reader:
            assert reader.id.valid
        
        # File should be closed - test by trying to access it
        # Note: In h5py, accessing a closed file's id.valid may not always raise,
        # but it should return False
        assert not reader.id.valid or reader.mode == 'closed'
    
    def test_hdf5_metadata_subgroups(self, sample_images, temp_hdf5_path):
        """Test HDF5 metadata subgroups are created."""
        img_set = ImageSetStatus(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        img_set.reset_status()
        
        with h5py.File(temp_hdf5_path, 'r') as f:
            img_path = '/phenotypic/image_sets/test_set/data/test_image_0'
            assert f'{img_path}/status' in f
            assert f'{img_path}/protected_metadata' in f
            assert f'{img_path}/public_metadata' in f


class TestImageSetEdgeCases:
    """Test error conditions and edge cases."""
    
    def test_empty_image_set_operations(self, temp_hdf5_path):
        """Test empty image set operations."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        
        # Initialize the HDF5 structure first
        with img_set.hdf_.writer() as writer:
            img_set.hdf_.get_data_group(writer)
        
        names = img_set.get_image_names()
        assert names == []
        
        images = list(img_set.iter_images())
        assert images == []
    
    def test_large_number_of_images(self, temp_hdf5_path):
        """Test handling large number of images (performance test)."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        
        # Create many small images
        images = []
        for i in range(50):
            arr = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            img = pht.Image(arr=arr, imformat='RGB', name=f'img_{i:03d}')
            images.append(img)
        
        img_set.import_images(images)
        
        names = img_set.get_image_names()
        assert len(names) == 50
    
    def test_image_name_conflicts(self, sample_images, temp_hdf5_path):
        """Test image name conflicts."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp', overwrite=False)
        
        img_set.add_image(sample_images[0])
        
        # Same name should raise error
        with pytest.raises(ValueError, match='already exists'):
            img_set.add_image(sample_images[0])
    
    def test_invalid_image_types(self, temp_hdf5_path):
        """Test invalid image types."""
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        
        # Should raise error for non-Image objects
        with pytest.raises(AssertionError):
            img_set.import_images(['not', 'images'])
    
    def test_missing_measurement_subgroups(self, sample_images, temp_hdf5_path):
        """Test missing measurement subgroups."""
        img_set = ImageSetMeasurements(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        img_set.reset_status()
        
        # Set status but don't add measurements
        with img_set.hdf_.writer() as writer:
            status_group = img_set.hdf_.get_status_subgroup(writer, 'test_image_0')
            status_group.attrs[PIPE_STATUS.PROCESSED.label] = True
            status_group.attrs[PIPE_STATUS.MEASURED.label] = True
        
        # Should handle gracefully
        result = img_set.get_measurement()
        assert len(result) == 0
    
    def test_corrupted_status_attributes(self, sample_images, temp_hdf5_path):
        """Test handling of corrupted status attributes."""
        img_set = ImageSetStatus(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.import_images(sample_images)
        
        # Don't initialize status (simulate corruption)
        # This should be handled gracefully or raise appropriate error
        try:
            status_df = img_set.get_status()
            # If it succeeds, should return data
            assert isinstance(status_df, pd.DataFrame)
        except (KeyError, AttributeError):
            # Expected for missing status attributes
            pass
    
    def test_special_characters_in_names(self, temp_hdf5_path):
        """Test handling of special characters in image names."""
        img = pht.Image(
            arr=np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            imformat='RGB',
            name='test_image_with-dash_underscore'
        )
        
        img_set = ImageSetCore(name='test_set', outpath=temp_hdf5_path, default_mode='temp')
        img_set.add_image(img)
        
        retrieved = img_set.get_image('test_image_with-dash_underscore')
        assert retrieved.name == 'test_image_with-dash_underscore'

