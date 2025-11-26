from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table, regionprops
from typing import List

from phenotypic.tools.constants_ import OBJECT, METADATA, IMAGE_TYPES, BBOX


class ObjectsAccessor:
    """Provide access to detected microbial colonies and their properties in agar plate images.

    This accessor enables researchers to analyze individual colonies in arrayed microbial cultures
    after colony detection has been performed. It provides methods for accessing colony labels,
    retrieving colony properties (area, intensity, position), extracting individual colony crops,
    and organizing colony data for high-throughput phenotypic screening workflows.

    The accessor operates on labeled object maps where each pixel value indicates which colony it
    belongs to (0 for background, 1+ for individual colonies). Properties are computed using
    scikit-image's regionprops functionality, providing standardized measurements for each colony.

    Note:
        This accessor can only be used after an ObjectDetector has been applied to the Image to
        identify and label individual colonies. Attempting to access before detection raises
        NoObjectsError.

    Attributes:
        _root_image (Image): The parent Image containing the labeled colony map (objmap).

    Examples:
        Access detected colonies and measure their properties:

        ```python
        from phenotypic import Image
        from phenotypic.detect import GitterDetector

        # Load plate image and detect colonies
        plate = Image.from_file("colony_array.png")
        detector = GitterDetector()
        detector.apply(plate)

        # Access colony properties
        print(f"Detected {len(plate.objects)} colonies")

        # Iterate over all colonies
        for colony in plate.objects:
            print(f"Colony area: {colony.gray.sum()}")

        # Get information for all colonies
        colony_info = plate.objects.info()
        print(colony_info[["ObjectLabel", "Bbox_CenterRR", "Bbox_CenterCC"]])
        ```
    """

    def __init__(self, root_image: Image):
        """Initialize the ObjectsAccessor with a parent Image.

        This method is called automatically when accessing the `objects` property of an Image.
        Users should not instantiate this class directly; instead, access it through the
        Image.objects property after applying an ObjectDetector.

        Args:
            root_image (Image): The parent Image containing detected colonies. Must have an
                objmap (object map) populated by an ObjectDetector that has been applied to
                identify and label individual colonies.

        Examples:
            Accessor is created automatically when accessing colonies:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("plate.png")
            detector = GitterDetector()
            detector.apply(plate)

            # ObjectsAccessor is automatically initialized
            accessor = plate.objects  # Uses __init__ internally
            print(f"Found {len(accessor)} colonies")
            ```
        """
        self._root_image = root_image

    def __len__(self) -> int:
        """Return the number of detected colonies in the plate image.

        This enables using Python's built-in len() function to quickly check how many colonies
        were detected by the object detector. Useful for quality control and validating that
        colony detection worked as expected.

        Returns:
            int: The total number of labeled colonies in the object map. Returns 0 if no
                colonies have been detected.

        Examples:
            Check colony count after detection:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("96well_array.png")
            detector = GitterDetector()
            detector.apply(plate)

            # Check if expected number of colonies detected
            colony_count = len(plate.objects)
            expected_count = 96
            if colony_count != expected_count:
                print(f"Warning: Expected {expected_count} colonies, found {colony_count}")
            ```

            Use in conditional logic:

            ```python
            if len(plate.objects) == 0:
                raise RuntimeError("No colonies detected. Check detector parameters.")
            ```
        """
        return self._root_image.num_objects

    def __iter__(self) -> Generator[Image, Any, None]:
        """Yield each object as an :class:`phenotypic.Image` crop.

        Usage:
            >>> for obj_image in image.objects:
            ...     print(obj_image.metadata["ImageName"])

        Yields:
            Image: Cropped image aligned to one labeled object at a time.

        Examples:
            >>> names = [obj.metadata[METADATA.IMAGE_NAME] for obj in image.objects]
        """
        for i in range(self._root_image.num_objects):
            yield self[i]

    def __getitem__(self, index: int) -> Image:
        """Return a cropped object image given its positional index.

        Usage:
            >>> first_object = image.objects[0]

        Args:
            index (int): Zero-based index of the desired labeled object.

        Returns:
            Image: A copy of the parent image cropped to the object's bounding
            box with metadata updated to ``ImageType='Object'``.

        Raises:
            IndexError: If ``index`` is outside of the labeled object range.

        Examples:
            >>> obj_img = image.objects[2]
            >>> obj_img.objmap.max()
            3
        """
        current_object = self.props[index]
        label = current_object.label
        object_image = self._root_image[current_object.slice]
        object_image.metadata[METADATA.IMAGE_TYPE] = IMAGE_TYPES.OBJECT.value
        object_image.objmap[object_image.objmap[:] != label] = 0
        return object_image

    @property
    def props(self):
        """List `skimage.measure.RegionProperties` objects for every label.

        Usage:
            >>> props = image.objects.props

        Returns:
            list[skimage.measure.RegionProperties]: Calculated region properties
            such as area, centroid, and bounding box for each object.

        Examples:
            >>> [prop.area for prop in image.objects.props]
            [315, 404, 289]
        """
        return regionprops(label_image=self._root_image.objmap[:], intensity_image=self._root_image.gray[:],
                           cache=False)

    @property
    def labels(self) -> List[int]:
        """Return all object labels detected in the image.

        Usage:
            >>> image.objects.labels

        Returns:
            list[int]: Numerical labels reported by :func:`skimage.measure.regionprops`.
            Returns an empty list if no objects are detected.

        Examples:
            >>> if 5 in image.objects.labels:
            ...     obj_five = image.objects.loc(5)
        """
        # considered using a simple numpy.unique() call on the object map, but wanted to guarantee that the labels will always be consistent
        # with any skimage outputs.
        return [x.label for x in self.props] if self.num_objects > 0 else []

    @property
    def slices(self):
        """Return bounding-box slices for each object.

        Usage:
            >>> rr_slice, cc_slice = image.objects.slices[0]

        Returns:
            list[slice | tuple[slice, ...]]: Region slices compatible with
            NumPy indexing to isolate each object.

        Examples:
            >>> cropped = image.gray[image.objects.slices[1]]
        """
        return [x.slice for x in self.props]

    def get_label_idx(self, object_label):
        """Return the positional index for a given object label.

        Usage:
            >>> idx = image.objects.get_label_idx(5)

        Args:
            object_label (int): Label id produced by the object detector and
                stored in ``image.objmap``.

        Returns:
            int: Zero-based index corresponding to ``labels`` ordering.

        Raises:
            IndexError: If ``object_label`` is not present in ``labels``.

        Examples:
            >>> image.objects.slices[image.objects.get_label_idx(3)]
        """
        return np.where(self.labels == object_label)[0][0]

    @property
    def num_objects(self) -> int:
        """Return the number of unique labels in the object map.

        Usage:
            >>> assert image.objects.num_objects == len(image.objects)

        Returns:
            int: Unique labels currently tracked in ``objmap``.
        """
        return self._root_image.num_objects

    def reset(self):
        """Reset the object map so the entire image becomes the target again.

        Usage:
            >>> image.objects.reset()

        Returns:
            None: This method mutates the parent image in-place.

        Examples:
            >>> image.objects.reset()
            >>> image.num_objects
            0
        """
        self._root_image.objmap.reset()

    def iloc(self, index: int) -> Image:
        """Return an object crop by integer position, similar to ``__getitem__``.

        Usage:
            >>> obj = image.objects.iloc(0)

        Args:
            index (int): Positional index of the object crop.

        Returns:
            Image: Cropped object image.

        Raises:
            IndexError: If ``index`` is outside the valid range.

        Examples:
            >>> image.objects.iloc(0).metadata[METADATA.IMAGE_TYPE]
            'Object'
        """
        return self._root_image[self.props[index].slice]

    def loc(self, label_number) -> Image:
        """Return an object crop using its label value.

        Usage:
            >>> nucleus = image.objects.loc(5)

        Args:
            label_number (int): Label assigned in ``objmap``.

        Returns:
            Image: Cropped bounding box for the specified label.

        Raises:
            IndexError: If ``label_number`` does not exist in the map.

        Examples:
            >>> obj = image.objects.loc(image.objects.labels[0])
        """
        idx = self.get_label_idx(label_number)
        return self._root_image[self.props[idx].slice]

    def info(self, include_metadata=True) -> pd.DataFrame:
        """Tabulate label, centroid, and bounding-box data for each object.

        Usage:
            >>> image.objects.info()

        Args:
            include_metadata (bool, optional): If True, prepend metadata columns
                via :meth:`phenotypic.core._image_parts.accessors.MetadataAccessor.insert_metadata`.
                Defaults to True.

        Returns:
            pandas.DataFrame: Table containing label, centroid row/column, and
            bounding-box coordinates for each object (plus metadata columns when
            requested).

        Examples:
            >>> info = image.objects.info()
            >>> info[["ObjectLabel", "Bbox_CenterRR"]]
        """
        info = pd.DataFrame(
                data=regionprops_table(
                        label_image=self._root_image.objmap[:],
                        properties=['label', 'centroid', 'bbox'],
                ),
        ).rename(columns={
            'label'     : OBJECT.LABEL,
            'centroid-0': str(BBOX.CENTER_RR),
            'centroid-1': str(BBOX.CENTER_CC),
            'bbox-0'    : str(BBOX.MIN_RR),
            'bbox-1'    : str(BBOX.MIN_CC),
            'bbox-2'    : str(BBOX.MAX_RR),
            'bbox-3'    : str(BBOX.MAX_CC),
        },
        )
        if include_metadata:
            return self._root_image.metadata.insert_metadata(info)
        else:
            return info

    def labels2series(self) -> pd.Series:
        """Create a labeled :class:`pandas.Series` of object labels.

        Usage:
            >>> labels = image.objects.labels2series()

        Returns:
            pandas.Series: Series named ``ObjectLabel`` listing each label.

        Examples:
            >>> labels = image.objects.labels2series()
            >>> measurements.join(labels, rsuffix="_idx")
        """
        labels = self.labels
        return pd.Series(
                data=labels,
                index=range(len(labels)),
                name=OBJECT.LABEL,
        )

    def relabel(self):
        """Recompute labels so that connected objects receive sequential ids.

        Usage:
            >>> image.objects.relabel()

        Returns:
            None: Labels are updated in-place on the parent ``objmap``.

        Examples:
            >>> image.objects.relabel()
            >>> image.objects.labels
            [1, 2, 3]
        """
        self._root_image.objmap.relabel()
