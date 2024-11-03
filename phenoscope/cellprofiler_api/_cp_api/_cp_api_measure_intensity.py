import logging
formatter = logging.Formatter(fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
							  datefmt='%m/%d/%Y %I:%M:%S')
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

# ----- Module Import -----

import pandas as pd

from cellprofiler.modules.measureobjectintensity import MeasureObjectIntensity


# ----- Pkg Relative Import -----
from ._cp_api_measure_areashape import CellProfilerApiMeasureAreaShape

# ----- Global Constants -----
INTEGRATED_INTENSITY = "IntegratedIntensity"
MEAN_INTENSITY = "MeanIntensity"
STD_INTENSITY = "StdIntensity"
MIN_INTENSITY = "MinIntensity"
MAX_INTENSITY = "MaxIntensity"
MEDIAN_INTENSITY = "MedianIntensity"
MAD_INTENSITY = "MADIntensity"
UPPER_QUARTILE_INTENSITY = "UpperQuartileIntensity"
LOWER_QUARTILE_INTENSITY = "LowerQuartileIntensity"

METRIC_LABEL = "Metric"


# ----- Main Class Definition -----
# Cleanup & optimize setting calls
class CellProfilerApiMeasureIntensity(CellProfilerApiMeasureAreaShape):
	status_intensity = True

	def measure_intensity(self, object_name=None, image_name=None):
		try:
			return self._measure_intensity(object_name=object_name, image_name=image_name)
		except KeyboardInterrupt:
			raise KeyboardInterrupt
		except:
			log.warning(f"Could not measure intensity for {self.image_name}", exc_info=True)
			self.status_validity = False
			return None

	def _measure_intensity(self, object_name=None, image_name=None):
		'''
		:return:
		'''
		if object_name is None:
			object_name = self.colony_name

		if image_name is None:
			image_name = self.image_name
		MEASUREMENT_CLASS_LABEL = "Intensity"
		mod = MeasureObjectIntensity()
		mod.images_list.value = image_name
		mod.objects_list.value = object_name
		self.pipeline.add_module(mod)
		mod.run(self.workspace)
		keys = self._get_feature_keys(object_name,mod)
		self.keys[f"{MEASUREMENT_CLASS_LABEL}"] = keys
		results = self._get_results(
			object_name, keys
		).reset_index(drop=False)

		metadata = results["Metric"].str.replace(image_name, "source")
		metadata = metadata.str.split("_", expand=True)
		metadata = metadata.rename(
			columns={
				0: "origin_method",
				1: "feature",
				2: "meta1",
				3: "meta2"
			}
		)
		metadata1 = metadata[metadata["origin_method"] == "Intensity"]
		metadata1 = metadata1["origin_method"] + "_" \
					+ metadata1["feature"]

		metadata2 = metadata[metadata["origin_method"] == "Location"]
		metadata2 = metadata2["origin_method"] + "_" \
					+ metadata2["feature"] + "_" \
					+ metadata2["meta1"]
		metadata = pd.concat([metadata1, metadata2], axis=0)
		results["Metric"] = metadata
		self._update_results(f"{MEASUREMENT_CLASS_LABEL}", results.set_index("Metric"))
		return self.results[f"{MEASUREMENT_CLASS_LABEL}"]
