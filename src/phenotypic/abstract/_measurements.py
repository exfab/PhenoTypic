import pandas as pd


# TODO: Not fully integrated yet
class Measurements:
    """A measurement container to hold image measurements for an image that is returned from MeasureFeature classes after measuring an image."""

    def __init__(self, name: str, image_name: str, measurement: pd.DataFrame):
        """
        Represents an object with a name, associated image name, and a dataframe
        for measurements. This class initializes the core attributes required
        for handling and processing data related to these entities.

        Args:
            name: The name of the object.
            image_name: The name of the associated image
            measurement: A pandas DataFrame containing measurement data.
        """
        self.name = name
        self.image_name = image_name
        self.table = measurement
