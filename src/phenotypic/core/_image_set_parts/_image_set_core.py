from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING: from phenotypic import Image

import os
import posixpath
from pathlib import Path

from typing import List
import h5py
from os import PathLike

from ._image_set_accessors._image_set_measurements_accessor import SetMeasurementAccessor
from phenotypic.util.constants_ import IO
from phenotypic.util import HDF


class ImageSetCore:
    """
    Handles the management and bulk processing of an image set, including importing from
    various sources, storing into an HDF5 file, and managing images efficiently.

    The `ImageSetCore` class facilitates large-scale image operations by importing images
    from either an in-memory list or a specified source directory/HDF5 file, storing the
    images into an output HDF5 file, and providing methods to manage and query the image set.
    It supports overwriting of existing datasets and ensures proper handling of HDF5 file
    groups for structured storage.

    Notes:
        - for developers: open a new writer in each function in order to prevent and data corruption with the hdf5 file

    Attributes:
        name (str): Name of the image set used for identification and structured storage.
        _src_path (Path | None): Path to the source directory or HDF5 file containing images.
            Initialized as a `Path` object or None if `image_list` is used.
        _out_path (Path): Path to the output HDF5 file storing the image set. Initialized
            as a `Path` object and defaults to the current working directory if not specified.
        _overwrite (bool): Indicates whether to overwrite existing data in the output HDF5 file.
        _hdf5_set_group_key (str): The group path in the HDF5 file where the image set is stored.
    """

    def __init__(self,
                 name: str,
                 image_template: Image | None = None,
                 src: List[Image] | PathLike | str | None = None,
                 outpath: PathLike | str | None = None,
                 overwrite: bool = False, ):
        """
        Initializes an image set for bulk processing of images.

        This constructor is responsible for setting up an image set by either importing
        images from a provided list, an HDF5 file, or a directory. It also handles the
        storing of the images into an output HDF5 file, with options for overwriting
        existing data. The `src` parameter automatically detects the input type.

        Args:
            name (str): The name of the image set to initialize.
            image_template: (Image | None): The Image object with settings to be used when constructing the Image.
                Can be a GridImage with ncols and nrows specified. Default is a 96-Plate GridImage
            src (List[Image] | PathLike | None, optional): The source for images. Can be:
                - A list of Image objects for importing from in-memory images
                - A PathLike object pointing to a source directory or HDF5 file containing images
                - None to connect to the output HDF5 file only
            outpath (PathLike | None, optional): The output HDF5 file where
                the image set will be stored. Defaults to the current working directory.
            overwrite (bool): Determines whether to overwrite existing data in the output
                HDF5 file. Defaults to False.

        Raises:
            ValueError: If no images or image sections are found in the provided `src` path.
            ValueError: If `src` is not a list of `Image` objects or a valid path.
        """
        import phenotypic
        self.name = name

        assert isinstance(image_template, (phenotypic.Image, type(None))), "image_template must be an Image object or None."
        self.image_template = image_template if image_template else phenotypic.GridImage(ncols=12, nrows=8)

        # Automatically detect the type of src parameter
        image_list = None
        src_path = None

        if src is not None:
            if isinstance(src, list):
                # src is a list of images
                image_list = src
            else:
                # src is a path-like object
                src_path = src

        src_path, outpath = Path(src_path) if src_path else None, Path(outpath) if outpath else Path.cwd() / f'{self.name}.hdf5'
        if outpath.is_dir(): outpath = outpath / f'{self.name}.hdf5'

        self.name, self._src_path, self._out_path = str(name), src_path, outpath
        self.hdf_ = HDF(filepath=outpath, name=self.name, mode='set')

        self._overwrite = overwrite

        # Define hdf5 group paths
        self._hdf5_parent_group_key: str = HDF.IMAGE_SET_ROOT_POSIX
        self._hdf5_set_group_key: str = posixpath.join(self._hdf5_parent_group_key, self.name)

        # Reminder: SetMeasurementAccessor are stored with each image
        #   self._hdf5_images_group_key/images/<image_name>/measurements <- that image's measurements
        self._hdf5_images_group_key: str = posixpath.join(self._hdf5_set_group_key, 'images')

        # If input source path handling
        if src_path:
            # If input path is an hdf5 file

            # If src and out are the same
            if (src_path.is_file()) and (src_path == outpath):
                pass

            # If src and out are different
            elif src_path.is_file() and src_path.suffix == '.h5':
                with h5py.File(src_path, mode='a', libver='latest') as src_filehandler, self.hdf_.safe_writer() as writer:
                    src_parent_group = self._get_hdf5_group(src_filehandler, self)
                    out_parent_group = self.hdf_.get_root_group(writer)

                    #   if the image set name is in the ImageSet group, copy the images over
                    if self.name in src_parent_group:
                        src_group = self._get_hdf5_group(src_filehandler, self._hdf5_set_group_key)

                        # overwrite if overwrite is true
                        if self.name in out_parent_group and overwrite is True: del out_parent_group[self.name]

                        # Should place a copy of the src_group into the parent group
                        src_filehandler.copy(src_group, out_parent_group, name=self.name, shallow=False)

                    #   else import all the images from Image section
                    elif HDF.SINGLE_IMAGE_ROOT_POSIX in src_filehandler:
                        src_image_group = self._get_hdf5_group(src_filehandler, self._hdf5_parent_group_key)

                        # overwrite if overwrite is true
                        if self.name in out_parent_group and overwrite is True: del out_parent_group[self.name]
                        src_filehandler.copy(src_image_group, out_parent_group, name=self.name, shallow=False)

                    else:
                        raise ValueError(f'No ImageSet named {self.name} or Image section found in {src_path}')

            # src_path image is a directory handling
            # only need out handler
            elif src_path.is_dir():
                image_filenames = [x for x in os.listdir(src_path) if x.endswith(IO.ACCEPTED_FILE_EXTENSIONS)]
                image_filenames.sort()
                with self.hdf_.safe_writer() as writer:
                    out_parent_group = self.hdf_.get_root_group(writer)

                    # Overwrite handling
                    if self.name in out_parent_group and overwrite is True: del out_parent_group[self.name]
                    images_group = self.hdf_.get_data_group(writer)

                    for fname in image_filenames:
                        image = self.image_template.imread(src_path / fname)
                        image._save_image2hdfgroup(grp=images_group, compression="gzip", compression_opts=4)

        # Image list handling
        # Only need out handler for this
        elif isinstance(image_list, list):
            assert all(isinstance(x, phenotypic.Image) for x in image_list), 'image_list must be a list of Image objects.'
            with self.hdf_.safe_writer() as writer:
                out_group = self._get_hdf5_group(writer, self._hdf5_parent_group_key)

                # Overwrite the data in the output folder
                if self.name in out_group and overwrite is True: del out_group[self.name]
                images_group = self.hdf_.get_data_group(writer)

                for image in image_list:
                    image._save_image2hdfgroup(grp=images_group, compression="gzip", compression_opts=4)
        elif not src_path and not image_list:  # connect to outpath hdf5 file only
            pass
        else:
            raise ValueError('image_list must be a list of Image objects or src_path must be a valid hdf5 file.')

    def _add_image2group(self, group, image: Image, overwrite: bool):
        """Helper function to add an image to a group that allows for reusing file handlers"""
        if image.name in group and overwrite is False:
            raise ValueError(f'Image named {image.name} already exists in ImageSet {self.name}.')
        else:
            image._save_image2hdfgroup(grp=group, compression="gzip", compression_opts=4)

    def add_image(self, image: Image, overwrite: bool | None = None):
        """
        Adds an image to an HDF5 file within a specified group.

        This method writes the provided image to an HDF5 file under a specified group.
        If the `overwrite` flag is set to True, the image will replace an existing
        dataset with the same name in the group. If set to False and a dataset with the
        same name already exists, the method will raise a ValueError.

        Args:
            image (Image): The image object to be added to the HDF5 group.
            overwrite (bool, optional): Indicates whether to overwrite an existing
                dataset if one with the same name exists. Defaults to None. If None, the method uses the
                initial overwrite value used when the class was created

        Raises:
            ValueError: If the `overwrite` flag is set to False and the image name is already in the ImageSet
        """
        with self.hdf_.writer() as writer:
            set_group = self.hdf_.get_group(writer, self._hdf5_images_group_key)
            self._add_image2group(group=set_group, image=image, overwrite=overwrite if overwrite else self._overwrite)

    @staticmethod
    def _get_hdf5_group(handler, name):
        name = str(name)
        if name in handler:
            return handler[name]
        else:
            return handler.create_group(name)

    def get_image_names(self) -> List[str]:
        """
        Retrieves the names of all images stored within the specified HDF5 group.

        This method opens an HDF5 file in read mode, accesses the specific group defined
        by the class's `_hdf5_set_group_key`, and retrieves the keys within that group,
        which represent the names of stored images.

        Returns:
            List[str]: A list of image names present in the specified HDF5 group.
        """
        with h5py.File(self._out_path, mode='r') as out_handler:
            set_group = self.hdf_.get_data_group(out_handler)
            names = list(set_group.keys())
        return names

    def get_image(self, image_name: str) -> Image:
        with self.hdf_.reader() as reader:
            image_group = self.hdf_.get_data_group(reader)
            if image_name in image_group:
                return self.image_template._load_from_hdf5_group(image_group[image_name])
            else:
                raise ValueError(f'Image named {image_name} not found in ImageSet {self.name}.')

    def get_measurement(self, image_name: str) -> pd.DataFrame:
        with h5py.File(self._out_path, mode='r', libver='latest', swmr=True) as reader:
            images_group = reader[str(self._hdf5_images_group_key)]
            if image_name in images_group:
                image_group = images_group[image_name]
                if IO.IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY in image_group:
                    return SetMeasurementAccessor._load_dataframe_from_hdf5_group(image_group)
                else:
                    raise ValueError(f'Image named {image_name} does not have a measurement.')
            else:
                raise ValueError(f'Image named {image_name} not found in ImageSet {self.name}.')

    def iter_images(self) -> iter:
        for image_name in self.get_image_names():
            with h5py.File(self._out_path, mode='r', libver='latest', swmr=True) as out_handler:
                image_group = self._get_hdf5_group(out_handler, posixpath.join(self._hdf5_images_group_key, image_name))
                image = self.image_template._load_from_hdf5_group(image_group)
            yield image
    
    def clear_hdf5_lock(self) -> bool:
        """
        Utility method to clear HDF5 file consistency flags that may prevent file access.
        
        This method attempts to clear HDF5 consistency flags using the h5clear utility,
        which can resolve "file is already open for write/SWMR write" errors.
        
        Returns:
            bool: True if the lock was successfully cleared, False otherwise.
            
        Example:
            >>> imageset = ImageSet(name='test', src='images/', outpath='output/')
            >>> if not imageset.clear_hdf5_lock():
            ...     print("Could not clear HDF5 lock. Try manually: h5clear -s output/test.h5")
        """
        import subprocess
        import os
        import logging
        
        logger = logging.getLogger(__name__)
        
        if not os.path.exists(self._out_path):
            logger.info(f"HDF5 file {self._out_path} does not exist, no lock to clear")
            return True
            
        try:
            logger.info(f"Attempting to clear HDF5 consistency flags for {self._out_path}")
            result = subprocess.run(['h5clear', '-s', str(self._out_path)], 
                                   capture_output=True, text=True, timeout=10)
            success = result.returncode == 0
            if success:
                logger.info("Successfully cleared HDF5 consistency flags")
            else:
                logger.warning(f"h5clear -s failed: {result.stderr}")
            
            return success
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Could not run h5clear: {e}")
            logger.info(f"To manually clear the lock, run: h5clear -s {self._out_path} && h5clear -f {self._out_path}")
            return False
    
    def apply(self, pipeline, inplace: bool = False, reset: bool = True, image_names: List[str] | str | None = None) -> 'ImageSet':
        """
        Apply a pipeline of operations to all images in the ImageSet.
        
        This method applies the given pipeline to each image in the ImageSet,
        with optional preallocation of measurement datasets if the pipeline
        contains measurements.
        
        Args:
            pipeline: An ImagePipeline object containing operations to apply
            inplace (bool): Whether to modify images in place. Defaults to False.
            reset (bool): Whether to reset images before applying pipeline. Defaults to True.
            image_names (List[str] | str | None): Specific images to process. If None, processes all images.
            
        Returns:
            ImageSet: Self for method chaining
            
        Example:
            >>> from phenotypic import ImagePipeline, CLAHE, OtsuDetector
            >>> pipeline = ImagePipeline([CLAHE(), OtsuDetector()])
            >>> imageset.apply(pipeline, inplace=True)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Determine which images to process
        if image_names is None:
            target_images = self.get_image_names()
        elif isinstance(image_names, str):
            target_images = [image_names]
        else:
            target_images = image_names
            
        logger.info(f"Applying pipeline to {len(target_images)} images in ImageSet '{self.name}'")
        
        # Preallocate measurement datasets if pipeline has measurements
        if hasattr(pipeline, '_measurements') and pipeline._measurements:
            logger.info("Pipeline contains measurements - preallocating datasets")
            self._preallocate_measurement_datasets_for_imageset(pipeline, target_images)
        
        # Apply pipeline to each image
        for image_name in target_images:
            try:
                image = self.get_image(image_name)
                processed_image = pipeline.apply(image, inplace=inplace, reset=reset)
                
                if not inplace:
                    # Save the processed image back to HDF5
                    with self.hdf_.writer() as handle:
                        image_group = self.hdf_.get_image_group(handle=handle, image_name=image_name)
                        processed_image._save_image2hdfgroup(image_group)
                        
                logger.debug(f"Successfully applied pipeline to image '{image_name}'")
                
            except Exception as e:
                logger.error(f"Failed to apply pipeline to image '{image_name}': {e}")
                # Mark image as ERROR in status
                self._mark_image_status(image_name, 'ERROR')
                continue
                
        return self
    
    def measure(self, pipeline, image_names: List[str] | str | None = None, include_metadata: bool = True) -> pd.DataFrame:
        """
        Apply measurements from a pipeline to all images in the ImageSet.
        
        This method applies the measurement components of the given pipeline to each
        image in the ImageSet, with preallocation of measurement datasets.
        
        Args:
            pipeline: An ImagePipeline object containing measurements to apply
            image_names (List[str] | str | None): Specific images to measure. If None, measures all images.
            include_metadata (bool): Whether to include image metadata in measurements. Defaults to True.
            
        Returns:
            pd.DataFrame: Consolidated measurements from all processed images
            
        Example:
            >>> from phenotypic import ImagePipeline, MeasureShape
            >>> pipeline = ImagePipeline(measurements=[MeasureShape()])
            >>> measurements = imageset.measure(pipeline)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Determine which images to process
        if image_names is None:
            target_images = self.get_image_names()
        elif isinstance(image_names, str):
            target_images = [image_names]
        else:
            target_images = image_names
            
        logger.info(f"Measuring {len(target_images)} images in ImageSet '{self.name}'")
        
        # Preallocate measurement datasets if pipeline has measurements
        if hasattr(pipeline, '_measurements') and pipeline._measurements:
            logger.info("Preallocating measurement datasets")
            self._preallocate_measurement_datasets_for_imageset(pipeline, target_images)
        
        all_measurements = []
        
        # Apply measurements to each image
        for image_name in target_images:
            try:
                image = self.get_image(image_name)
                measurements = pipeline.measure(image, include_metadata=include_metadata)
                
                # Add image name to measurements for identification
                measurements['image_name'] = image_name
                all_measurements.append(measurements)
                
                # Save measurements to HDF5
                with self.hdf_.writer() as handle:
                    image_group = self.hdf_.get_image_group(handle=handle, image_name=image_name)
                    measurement_group = self.hdf_.get_measurement_group(handle=handle, image_name=image_name)
                    self.measurements._save_dataframe_to_hdf5_group(measurements, measurement_group)
                    
                logger.debug(f"Successfully measured image '{image_name}'")
                
            except Exception as e:
                logger.error(f"Failed to measure image '{image_name}': {e}")
                # Mark image as ERROR in status
                self._mark_image_status(image_name, 'ERROR')
                continue
        
        # Consolidate all measurements into a single DataFrame
        if all_measurements:
            return pd.concat(all_measurements, ignore_index=True)
        else:
            logger.warning("No measurements were successfully collected")
            return pd.DataFrame()
    
    def _preallocate_measurement_datasets_for_imageset(self, pipeline, image_names: List[str]):
        """
        Preallocate measurement datasets for the given images and pipeline.
        
        This method creates empty HDF5 datasets for measurements that will be
        generated by the pipeline, enabling efficient parallel writing.
        
        Args:
            pipeline: Pipeline object containing measurements
            image_names: List of image names to preallocate datasets for
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Get a sample image to determine measurement structure
            if not image_names:
                logger.warning("No images provided for preallocation")
                return
                
            sample_image = self.get_image(image_names[0])
            
            # Generate sample measurements to determine DataFrame structure
            sample_measurements = pipeline.measure(sample_image, include_metadata=True)
            
            logger.info(f"Preallocating datasets for {len(image_names)} images based on sample measurements")
            
            # Preallocate datasets for each image
            with self.hdf_.writer() as handle:
                for image_name in image_names:
                    try:
                        measurement_group = self.hdf_.get_measurement_group(handle=handle, image_name=image_name)
                        self.measurements._preallocate_swmr_measurement_datasets(
                            measurement_group, sample_measurements, group_name="measurements"
                        )
                        logger.debug(f"Preallocated measurement datasets for image '{image_name}'")
                    except Exception as e:
                        logger.error(f"Failed to preallocate datasets for image '{image_name}': {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to preallocate measurement datasets: {e}")
            # Don't raise - allow processing to continue without preallocation
    
    def _mark_image_status(self, image_name: str, status: str):
        """
        Mark an image with the given status.
        
        Args:
            image_name: Name of the image
            status: Status to set (e.g., 'ERROR', 'PROCESSED', 'MEASURED')
        """
        try:
            with self.hdf_.writer() as handle:
                image_group = self.hdf_.get_image_group(handle=handle, image_name=image_name)
                image_group.attrs[status] = True
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to mark image '{image_name}' with status '{status}': {e}")
            return False
