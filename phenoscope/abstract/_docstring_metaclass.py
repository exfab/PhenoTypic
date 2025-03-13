
class MeasureDocstringMeta(type):
    """
    Metaclass to automatically set the docstring of measure() to match _operate()
    """

    def __new__(cls, name, bases, dct):
        # Set the docstring of measure() to match _operate()
        if '_operate' in dct and 'measure' in dct:
            dct['measure'].__doc__ = dct['_operate'].__doc__
        return super().__new__(cls, name, bases, dct)

class ApplyDocstringMeta(type):
    """
    Metaclass to automatically set the docstring of apply() to match _operate()
    """

    def __new__(cls, name, bases, dct):
        # Set the docstring of measure() to match _operate()
        if '_operate' in dct and 'measure' in dct:
            dct['measure'].__doc__ = dct['_operate'].__doc__
        return super().__new__(cls, name, bases, dct)
