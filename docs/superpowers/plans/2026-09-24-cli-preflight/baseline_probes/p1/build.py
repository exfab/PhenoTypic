import sys; sys.path.insert(0, sys.argv[1])
from my_custom_ops import MyThreshDetector
from phenotypic import ImagePipeline
from phenotypic.measure import MeasureSize
p = ImagePipeline(ops={'det': MyThreshDetector(thresh=0.4)}, meas={'size': MeasureSize()})
p.to_json(sys.argv[2]); print("wrote", sys.argv[2])
print(open(sys.argv[2]).read()[:600])
