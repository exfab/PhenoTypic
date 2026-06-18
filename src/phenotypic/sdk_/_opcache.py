from dataclasses import dataclass
from enum import Enum


# TODO: Implement the rest of the OpCache and OpCacheKey logic

@dataclass
class OpCache:
    """A dataclass for storing references intermediate variables used in conjunction
    within classes that have `.inspect()` methods. `ImageOperation` and `MeasureFeature`
    classes should use this by having a private `self.__cache` attribute that points
    to an OpCache. The OpCache variables are defined within the `_operate()` method
    and referenced in the `inspect()` method. All value setting should be done through
    the `BaseOperation._setcache(key, value)` method, and retrievals through the
    `BaseOperation._getcache(key, value)` method. In pipeline settings, the
    `BaseOperation._retain_cache: bool` flag will be set to false, to avoid OOM errors
    that can occur from all the intermediates. In that scenario, the cache will be reset
    so all intermediates are dereferenced. """
    pass


class OpCacheKey(str, Enum):
    """This provides a structured way to store the labels for an OpCache for a class.
    This allows for utilizing the uniqueness of an Enum class, but being able to plug in
    the key values like a str. See the MeasurementInfo class for a similar implementation
    """
    pass
