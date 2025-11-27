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
        """Iterate over all detected colonies, yielding each as a cropped Image.

        This enables using the accessor in for loops and comprehensions to process each colony
        individually. Each yielded Image is cropped to the colony's bounding box and contains
        only that colony's pixels in its object map (all other pixels set to 0).

        This is particularly useful for batch processing colonies, computing per-colony metrics,
        or performing quality control by visually inspecting individual colonies.

        Yields:
            Image: A cropped Image for each colony, in order by position index (0 to N-1).
                Each cropped Image has its metadata updated with ImageType='Object' and
                contains only the pixels within that colony's bounding box. The objmap of
                each yielded Image contains only the current colony's label, with all other
                pixels set to 0.

        Examples:
            Compute average intensity for each colony:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("fluorescent_colonies.png")
            detector = GitterDetector()
            detector.apply(plate)

            # Calculate mean fluorescence for each colony
            intensities = []
            for colony in plate.objects:
                mean_intensity = colony.gray.mean()
                intensities.append(mean_intensity)

            print(f"Average colony intensity: {sum(intensities) / len(intensities)}")
            ```

            Filter colonies by size:

            ```python
            # Get all colonies larger than 500 pixels
            large_colonies = [
                colony for colony in plate.objects
                if (colony.objmap > 0).sum() > 500
            ]
            print(f"Found {len(large_colonies)} large colonies")
            ```

            Extract colony names for tracking:

            ```python
            colony_names = [
                obj.metadata[METADATA.IMAGE_NAME]
                for obj in plate.objects
            ]
            ```
        """
        for i in range(self._root_image.num_objects):
            yield self[i]

    def __getitem__(self, index: int) -> Image:
        """Extract a specific colony by its position index.

        This enables bracket notation (e.g., plate.objects[0]) to access individual colonies
        by their sequential position. The returned Image is cropped to the colony's bounding
        box and contains only that colony in its object map.

        This is useful when you need to access a specific colony by position, such as the first
        or last detected colony, or when iterating with enumerate() to track both index and colony.

        Args:
            index (int): Zero-based position index of the desired colony. Must be in the range
                [0, len(objects)-1]. Negative indexing is not supported.

        Returns:
            Image: A cropped Image containing only the specified colony. The Image is cropped to
                the colony's bounding box, has metadata updated to ImageType='Object', and has
                an objmap where pixels belonging to the colony retain their label value while
                all other pixels are set to 0.

        Raises:
            IndexError: If index is negative or >= the total number of colonies.

        Examples:
            Extract the first detected colony:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("colony_plate.png")
            detector = GitterDetector()
            detector.apply(plate)

            # Get first colony
            first_colony = plate.objects[0]
            print(f"First colony label: {first_colony.objmap.max()}")
            print(f"Colony size: {(first_colony.objmap > 0).sum()} pixels")
            ```

            Access specific colonies for comparison:

            ```python
            # Compare first and last colonies
            first = plate.objects[0]
            last = plate.objects[len(plate.objects) - 1]

            print(f"First colony mean intensity: {first.gray.mean()}")
            print(f"Last colony mean intensity: {last.gray.mean()}")
            ```

            Use with enumerate for indexed processing:

            ```python
            for idx, colony in enumerate(plate.objects):
                if idx % 10 == 0:
                    # Process every 10th colony
                    colony.show()
            ```
        """
        current_object = self.props[index]
        label = current_object.label
        object_image = self._root_image[current_object.slice]
        object_image.metadata[METADATA.IMAGE_TYPE] = IMAGE_TYPES.OBJECT.value
        object_image.objmap[object_image.objmap[:] != label] = 0
        return object_image

    @property
    def props(self) -> list:
        """Compute region properties for all detected colonies using scikit-image.

        This property provides access to scikit-image's RegionProperties objects, which contain
        detailed morphological and intensity measurements for each colony. Each RegionProperties
        object provides properties like area, perimeter, centroid, bounding box coordinates,
        mean intensity, and many others.

        The properties are computed dynamically each time this property is accessed (cache=False),
        ensuring measurements always reflect the current state of the object map and intensity image.

        Returns:
            list[skimage.measure.RegionProperties]: A list of RegionProperties objects, one per
                detected colony. Each object provides lazy access to properties like:
                - area: Number of pixels in the colony
                - centroid: (row, col) coordinates of the colony center
                - bbox: Bounding box as (min_row, min_col, max_row, max_col)
                - mean_intensity: Mean pixel intensity within the colony region
                - perimeter, eccentricity, solidity, and many more

                See scikit-image regionprops documentation for the complete list of available
                properties.

        Examples:
            Extract colony areas:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("colony_array.png")
            detector = GitterDetector()
            detector.apply(plate)

            # Get areas of all colonies
            areas = [prop.area for prop in plate.objects.props]
            print(f"Colony sizes: {areas}")
            print(f"Mean colony size: {sum(areas) / len(areas):.1f} pixels")
            ```

            Access multiple properties per colony:

            ```python
            for prop in plate.objects.props:
                print(f"Colony {prop.label}:")
                print(f"  Area: {prop.area} pixels")
                print(f"  Centroid: ({prop.centroid[0]:.1f}, {prop.centroid[1]:.1f})")
                print(f"  Mean intensity: {prop.mean_intensity:.2f}")
                print(f"  Eccentricity: {prop.eccentricity:.3f}")
            ```

            Filter colonies by morphology:

            ```python
            # Find circular colonies (low eccentricity)
            circular_colonies = [
                prop for prop in plate.objects.props
                if prop.eccentricity < 0.5
            ]
            print(f"Found {len(circular_colonies)} circular colonies")
            ```
        """
        return regionprops(label_image=self._root_image.objmap[:], intensity_image=self._root_image.gray[:],
                           cache=False)

    @property
    def labels(self) -> List[int]:
        """Get the unique label identifiers for all detected colonies.

        Each colony in the object map is assigned a unique positive integer label (typically
        starting from 1). This property returns all such labels, which can be used to select
        specific colonies with the loc() method or to verify which colonies are present.

        Labels are extracted from scikit-image's regionprops to ensure consistency with all
        property calculations, rather than using numpy.unique() on the object map directly.

        Returns:
            list[int]: A list of unique colony labels in ascending order. Returns an empty list
                if no colonies have been detected. Background pixels (labeled 0) are excluded.

        Examples:
            Check which colonies are present:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("colony_array.png")
            detector = GitterDetector()
            detector.apply(plate)

            labels = plate.objects.labels
            print(f"Detected colonies with labels: {labels}")
            print(f"Labels range from {min(labels)} to {max(labels)}")
            ```

            Check if a specific colony exists:

            ```python
            if 5 in plate.objects.labels:
                colony_5 = plate.objects.loc(5)
                print(f"Colony 5 area: {colony_5.gray.sum()}")
            else:
                print("Colony 5 not found")
            ```

            Verify contiguous labeling:

            ```python
            labels = plate.objects.labels
            expected_labels = list(range(1, len(labels) + 1))
            if labels == expected_labels:
                print("Labels are contiguous")
            else:
                print("Labels have gaps - consider using relabel()")
            ```
        """
        # considered using a simple numpy.unique() call on the object map, but wanted to guarantee that the labels will always be consistent
        # with any skimage outputs.
        return [x.label for x in self.props] if self.num_objects > 0 else []

    @property
    def slices(self) -> list:
        """Get bounding box slices for efficiently cropping to each colony region.

        This property returns NumPy-compatible slice objects that define the rectangular bounding
        box around each colony. These slices can be used directly with array indexing to extract
        the minimal rectangular region containing each colony, which is much more efficient than
        working with the full plate image when processing individual colonies.

        Each slice is a tuple of (row_slice, col_slice) that can be used with NumPy indexing
        syntax to extract the bounding box region from any array with the same dimensions.

        Returns:
            list[tuple[slice, slice]]: A list of slice tuples, one per colony, in the same order
                as the labels. Each tuple contains (row_slice, col_slice) where:
                - row_slice defines the row range: slice(min_row, max_row)
                - col_slice defines the column range: slice(min_col, max_col)

        Examples:
            Extract bounding box regions from the grayscale image:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("colony_array.png")
            detector = GitterDetector()
            detector.apply(plate)

            # Get the bounding box region for the first colony
            first_slice = plate.objects.slices[0]
            first_colony_region = plate.gray[first_slice]
            print(f"First colony bounding box shape: {first_colony_region.shape}")
            ```

            Process all colonies using their bounding boxes:

            ```python
            for idx, bbox_slice in enumerate(plate.objects.slices):
                colony_region = plate.gray[bbox_slice]
                mean_intensity = colony_region.mean()
                print(f"Colony {idx}: mean intensity = {mean_intensity:.2f}")
            ```

            Extract bounding boxes from multiple image channels:

            ```python
            # Process RGB channels for each colony
            for bbox_slice in plate.objects.slices:
                r_channel = plate.rgb[bbox_slice][..., 0]
                g_channel = plate.rgb[bbox_slice][..., 1]
                b_channel = plate.rgb[bbox_slice][..., 2]
                print(f"R: {r_channel.mean():.1f}, G: {g_channel.mean():.1f}, "
                      f"B: {b_channel.mean():.1f}")
            ```
        """
        return [x.slice for x in self.props]

    def get_label_idx(self, object_label: int) -> int:
        """Convert a colony label to its position index in the labels list.

        This method maps from label identifiers (which may have gaps or start from values other
        than 0) to position indices (which are always contiguous from 0 to N-1). This is useful
        when you know a colony's label but need its position for indexing into lists or arrays
        organized by position.

        Args:
            object_label (int): The label identifier to look up. This is the value stored in the
                objmap for pixels belonging to the desired colony, typically a positive integer
                starting from 1.

        Returns:
            int: The zero-based position index where this label appears in the labels list.
                This index can be used with methods like iloc(), slices[], or props[].

        Raises:
            IndexError: If object_label is not present in the current labels list. This can
                occur if the label doesn't exist or if colonies have been filtered/removed.

        Examples:
            Map label to index for accessing properties:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("colony_array.png")
            detector = GitterDetector()
            detector.apply(plate)

            # Find the position index for colony label 5
            idx = plate.objects.get_label_idx(5)
            print(f"Colony with label 5 is at position {idx}")

            # Use the index to access properties
            colony_slice = plate.objects.slices[idx]
            colony_props = plate.objects.props[idx]
            print(f"Colony 5 area: {colony_props.area}")
            ```

            Use with non-contiguous labels:

            ```python
            # After filtering, labels may not be contiguous
            labels = plate.objects.labels  # e.g., [1, 2, 5, 8, 10]

            # Get position index for label 8 (which is at position 3)
            idx = plate.objects.get_label_idx(8)  # Returns 3
            colony_8 = plate.objects.iloc(idx)
            ```

            Verify label exists before accessing:

            ```python
            desired_label = 42
            if desired_label in plate.objects.labels:
                idx = plate.objects.get_label_idx(desired_label)
                print(f"Found colony {desired_label} at index {idx}")
            else:
                print(f"Colony {desired_label} not found")
            ```
        """
        return np.where(self.labels == object_label)[0][0]

    @property
    def num_objects(self) -> int:
        """Get the total number of detected colonies.

        This property provides the count of unique colony labels in the object map. It is
        equivalent to len(image.objects) but provides a more explicit property-based interface.
        The count excludes background pixels (label 0).

        Returns:
            int: The number of unique colony labels currently tracked in the objmap. Returns 0
                if no colonies have been detected.

        Examples:
            Verify expected colony count:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("96well_plate.png")
            detector = GitterDetector()
            detector.apply(plate)

            # Check if detection found the expected number of colonies
            expected = 96
            actual = plate.objects.num_objects
            if actual != expected:
                print(f"Warning: Expected {expected} colonies, found {actual}")
            ```

            Verify consistency with len():

            ```python
            assert plate.objects.num_objects == len(plate.objects)
            ```
        """
        return self._root_image.num_objects

    def reset(self) -> None:
        """Clear all colony labels from the object map.

        This method resets the object map to its initial empty state by setting all pixels to 0,
        effectively removing all colony detections. After calling reset(), the image reverts to
        having no detected colonies (num_objects = 0) and the entire image becomes the analysis
        target again.

        This is useful when you want to re-run colony detection with different parameters or when
        you need to clear previous detection results before applying a different detection method.

        Returns:
            None: This method modifies the parent Image in-place by clearing its objmap. No value
                is returned.

        Examples:
            Clear detections to re-run with different parameters:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("colony_array.png")

            # First detection attempt
            detector1 = GitterDetector(threshold=0.5)
            detector1.apply(plate)
            print(f"First attempt: {plate.objects.num_objects} colonies")

            # Clear and try with different parameters
            plate.objects.reset()
            print(f"After reset: {plate.objects.num_objects} colonies")  # 0

            detector2 = GitterDetector(threshold=0.3)
            detector2.apply(plate)
            print(f"Second attempt: {plate.objects.num_objects} colonies")
            ```

            Verify reset clears all colonies:

            ```python
            plate.objects.reset()
            assert plate.num_objects == 0
            assert len(plate.objects.labels) == 0
            assert plate.objmap.max() == 0
            ```
        """
        self._root_image.objmap.reset()

    def iloc(self, index: int) -> Image:
        """Access a colony by position index using pandas-style iloc syntax.

        This method provides a pandas-like interface for accessing colonies by their position
        index. Unlike __getitem__ (which also uses position), iloc() returns a crop that preserves
        all labels in the objmap, while __getitem__ zeros out non-matching labels.

        Use iloc() when you need the minimal bounding box crop but want to preserve the original
        label structure, which can be useful for certain analysis workflows.

        Args:
            index (int): Zero-based position index of the colony to extract. Must be in the range
                [0, num_objects-1].

        Returns:
            Image: A cropped Image containing the bounding box region of the specified colony.
                Unlike __getitem__, this crop preserves all label values in the objmap without
                zeroing non-matching pixels. The metadata is updated with the parent's metadata
                but ImageType is not changed to 'Object'.

        Raises:
            IndexError: If index is negative or >= num_objects.

        Examples:
            Access first colony with pandas-style syntax:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("colony_array.png")
            detector = GitterDetector()
            detector.apply(plate)

            # Access first colony
            first = plate.objects.iloc(0)
            print(f"Bounding box shape: {first.shape}")
            ```

            Compare iloc() vs __getitem__:

            ```python
            # iloc preserves all labels in the crop
            crop_iloc = plate.objects.iloc(5)
            print(f"Labels in crop (iloc): {crop_iloc.objmap.max()}")  # May show label 5

            # __getitem__ zeros out non-matching labels
            crop_getitem = plate.objects[5]
            print(f"Labels in crop ([5]): {crop_getitem.objmap.max()}")  # Shows only label 5
            ```

            Extract multiple specific colonies:

            ```python
            # Get colonies at positions 0, 5, and 10
            selected_indices = [0, 5, 10]
            selected_colonies = [plate.objects.iloc(i) for i in selected_indices]
            ```
        """
        return self._root_image[self.props[index].slice]

    def loc(self, label_number: int) -> Image:
        """Access a colony by its label identifier using pandas-style loc syntax.

        This method provides a pandas-like interface for accessing colonies by their label value
        (rather than position index). This is useful when you know the specific label ID of a
        colony from the objmap and want to extract just that colony's bounding box region.

        Unlike iloc() which uses position, loc() uses the actual label value assigned in the
        object map. This is particularly helpful when labels are non-contiguous or when working
        with specific colonies identified by their label in previous analysis steps.

        Args:
            label_number (int): The label identifier assigned to the colony in the objmap.
                This is typically a positive integer (1, 2, 3, ...) but may have gaps if
                colonies were filtered or removed.

        Returns:
            Image: A cropped Image containing the bounding box region of the colony with the
                specified label. The crop preserves all label values in the objmap without
                zeroing non-matching pixels.

        Raises:
            IndexError: If label_number does not exist in the current labels list.

        Examples:
            Access colony by its label:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("colony_array.png")
            detector = GitterDetector()
            detector.apply(plate)

            # Access colony with label 5
            colony_5 = plate.objects.loc(5)
            print(f"Colony 5 bounding box: {colony_5.shape}")
            ```

            Access first colony by its label (not position):

            ```python
            # Get the label of the first colony
            first_label = plate.objects.labels[0]  # e.g., could be 1

            # Access by that label
            first_colony = plate.objects.loc(first_label)
            ```

            Use with non-contiguous labels:

            ```python
            # Labels might be: [1, 2, 5, 8, 10] after filtering
            labels = plate.objects.labels

            # Access colony with label 8 (not position 8)
            colony_8 = plate.objects.loc(8)

            # Compare with position-based access
            colony_at_pos_3 = plate.objects.iloc(3)  # Also gets label 8
            ```

            Safe access with label validation:

            ```python
            desired_label = 42
            if desired_label in plate.objects.labels:
                colony = plate.objects.loc(desired_label)
                print(f"Found colony {desired_label}")
            else:
                print(f"Colony {desired_label} not found")
            ```
        """
        idx = self.get_label_idx(label_number)
        return self._root_image[self.props[idx].slice]

    def info(self, include_metadata: bool = True) -> pd.DataFrame:
        """Generate a summary table of colony positions and bounding boxes.

        This method creates a pandas DataFrame containing key positional information for all
        detected colonies, including their labels, centroid coordinates, and bounding box
        coordinates. This is particularly useful for organizing colony data for downstream
        analysis, quality control, or exporting to other tools.

        The table includes one row per colony with columns for the colony label, centroid
        position (row and column), and bounding box coordinates (min/max row and column).
        Optionally, image metadata can be prepended to provide experimental context.

        Args:
            include_metadata (bool, optional): If True, prepend image metadata columns
                (such as ImageName, UUID, ImageType) to the DataFrame using the Image's
                metadata.insert_metadata() method. This adds experimental context to the
                colony information. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame with one row per colony containing:
                - ObjectLabel: The colony's label identifier
                - Bbox_CenterRR: Row coordinate of the colony centroid
                - Bbox_CenterCC: Column coordinate of the colony centroid
                - Bbox_MinRR: Minimum row coordinate of the bounding box
                - Bbox_MinCC: Minimum column coordinate of the bounding box
                - Bbox_MaxRR: Maximum row coordinate of the bounding box
                - Bbox_MaxCC: Maximum column coordinate of the bounding box

                If include_metadata=True, additional metadata columns are prepended.

        Examples:
            Get basic colony information:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("colony_array.png")
            detector = GitterDetector()
            detector.apply(plate)

            # Get colony information
            colony_info = plate.objects.info()
            print(colony_info.head())
            print(f"Columns: {colony_info.columns.tolist()}")
            ```

            Access specific columns:

            ```python
            # Get just labels and centroids
            labels_centroids = colony_info[["ObjectLabel", "Bbox_CenterRR", "Bbox_CenterCC"]]
            print(labels_centroids)
            ```

            Export colony positions for other tools:

            ```python
            # Get info without metadata for cleaner export
            colony_positions = plate.objects.info(include_metadata=False)
            colony_positions.to_csv("colony_positions.csv", index=False)
            ```

            Filter colonies by position:

            ```python
            info = plate.objects.info(include_metadata=False)

            # Find colonies in the upper half of the image
            upper_colonies = info[info["Bbox_CenterRR"] < 500]
            print(f"Found {len(upper_colonies)} colonies in upper half")
            ```

            Calculate bounding box dimensions:

            ```python
            info = plate.objects.info(include_metadata=False)

            # Calculate width and height of each colony's bounding box
            info["BBox_Width"] = info["Bbox_MaxCC"] - info["Bbox_MinCC"]
            info["BBox_Height"] = info["Bbox_MaxRR"] - info["Bbox_MinRR"]

            print(f"Average colony width: {info['BBox_Width'].mean():.1f} pixels")
            print(f"Average colony height: {info['BBox_Height'].mean():.1f} pixels")
            ```
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
        """Convert colony labels to a pandas Series for joining with measurement DataFrames.

        This method creates a pandas Series containing all colony labels with a properly named
        column ('ObjectLabel') and indexed by position. This is specifically designed to
        facilitate joining labels with measurement DataFrames that are indexed by position,
        which is a common pattern in high-throughput phenotypic analysis workflows.

        The returned Series can be easily joined or merged with other pandas DataFrames
        containing per-colony measurements to add label information.

        Returns:
            pd.Series: A pandas Series with:
                - data: The colony labels (integers)
                - index: Position indices (0 to N-1)
                - name: 'ObjectLabel' for proper DataFrame column naming

        Examples:
            Join labels with measurement data:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector
            from phenotypic.measure import AreaMeasurer
            import pandas as pd

            plate = Image.from_file("colony_array.png")
            detector = GitterDetector()
            detector.apply(plate)

            # Calculate measurements (indexed by position)
            measurer = AreaMeasurer()
            measurements = measurer.measure(plate)

            # Add labels to measurements
            labels = plate.objects.labels2series()
            measurements_with_labels = measurements.join(labels)

            print(measurements_with_labels.head())
            ```

            Merge with custom analysis results:

            ```python
            import numpy as np
            import pandas as pd

            # Custom analysis results indexed by position
            custom_analysis = pd.DataFrame({
                'mean_intensity': [125.3, 142.7, 98.1, ...],
                'max_intensity': [255, 243, 189, ...]
            }, index=range(len(plate.objects)))

            # Add colony labels
            labels_series = plate.objects.labels2series()
            analysis_with_labels = custom_analysis.join(labels_series)

            # Now can group or filter by label
            print(analysis_with_labels)
            ```

            Use as a lookup table:

            ```python
            labels_series = plate.objects.labels2series()

            # Get label for position 5
            label_at_pos_5 = labels_series.iloc[5]
            print(f"Colony at position 5 has label {label_at_pos_5}")
            ```

            Handle potential label suffix conflicts:

            ```python
            # If DataFrame already has an 'ObjectLabel' column
            labels = plate.objects.labels2series()
            measurements_with_labels = measurements.join(labels, rsuffix="_new")
            # Creates 'ObjectLabel' and 'ObjectLabel_new' columns
            ```
        """
        labels = self.labels
        return pd.Series(
                data=labels,
                index=range(len(labels)),
                name=OBJECT.LABEL,
        )

    def relabel(self) -> None:
        """Renumber colony labels to be sequential and contiguous.

        This method relabels all colonies in the object map so that labels are sequential
        integers starting from 1 (i.e., 1, 2, 3, ..., N for N colonies). This is useful
        after filtering or removing colonies, which can leave gaps in the label sequence,
        or when you need consistent, predictable label numbering for downstream analysis.

        The relabeling preserves the spatial relationships between colonies but assigns new
        labels in a consistent order. Connected components (colonies) are assigned sequential
        IDs based on their position in the label image.

        Returns:
            None: This method modifies the parent Image's objmap in-place. Labels are
                renumbered directly in the object map, and no value is returned.

        Examples:
            Relabel after filtering to remove gaps:

            ```python
            from phenotypic import Image
            from phenotypic.detect import GitterDetector

            plate = Image.from_file("colony_array.png")
            detector = GitterDetector()
            detector.apply(plate)

            # Before relabeling (may have gaps)
            print(f"Labels before: {plate.objects.labels}")
            # e.g., [1, 2, 5, 7, 10, 15]

            # Relabel to make sequential
            plate.objects.relabel()

            # After relabeling (sequential)
            print(f"Labels after: {plate.objects.labels}")
            # e.g., [1, 2, 3, 4, 5, 6]
            ```

            Ensure consistent labeling for reproducibility:

            ```python
            # After any colony filtering or modification
            plate.objects.relabel()

            # Now labels are guaranteed to be 1, 2, 3, ..., N
            assert plate.objects.labels == list(range(1, len(plate.objects) + 1))
            ```

            Relabel after manual objmap modifications:

            ```python
            # After custom filtering or editing the objmap
            import numpy as np

            # Remove small colonies (custom filtering)
            for prop in plate.objects.props:
                if prop.area < 100:
                    mask = plate.objmap[:] == prop.label
                    plate.objmap[:][mask] = 0

            # Relabel to clean up the label sequence
            plate.objects.relabel()
            print(f"Remaining colonies: {len(plate.objects)}")
            print(f"New labels: {plate.objects.labels}")
            ```

            Compare before and after relabeling:

            ```python
            labels_before = plate.objects.labels.copy()
            plate.objects.relabel()
            labels_after = plate.objects.labels

            print(f"Before: {labels_before}")
            print(f"After: {labels_after}")
            print(f"Same count: {len(labels_before) == len(labels_after)}")
            ```
        """
        self._root_image.objmap.relabel()
