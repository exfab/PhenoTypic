"""Self-registering variant: import side effect exposes class on phenotypic."""
import phenotypic
from my_custom_ops import MyThreshDetector
phenotypic.MyThreshDetector = MyThreshDetector
