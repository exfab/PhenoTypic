import logging
formatter = logging.Formatter(fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S')
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

# ----- Module Import -----
from cellprofiler.modules.measureobjectsizeshape import MeasureObjectSizeShape

# ----- Pkg Relative Import -----
from ._cp_api_object import CellProfilerApiObject

# ----- Global Constants -----
METRIC_LABEL = "Metric"

MAIN_METRICS = [
    ""
]


# ----- Main Class Definition -----
# TODO: Cleanup & optimize setting calls
class CellProfilerApiMeasureAreaShape(CellProfilerApiObject):
    status_areashape = True
    areashape_calculate_adv = False
    areashape_calculate_zernikes = False

    def measure_areashape(self, object_name=None, calculate_adv=False, calculate_zernike=False):
        try:
            return self._measure_areashape(object_name=object_name,
                                           calculate_adv=calculate_adv,
                                           calculate_zernike=calculate_zernike)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            log.warning(f"Could not measure AreaShape for {self.image_name}", exc_info=True)
            self.status_validity = False
            return None

    def _measure_areashape(self, object_name=None, calculate_adv=False, calculate_zernike=False):
        if object_name is None:
            object_name = f"{self.colony_name}"
        MEASUREMENT_CLASS_LABEL = "AreaShape"
        log.debug(f"Measuring Area Shape of sample {object_name}")
        mod = MeasureObjectSizeShape()
        mod.calculate_advanced.value = calculate_adv
        mod.calculate_zernikes.value = calculate_zernike
        mod.objects_list.value = f"{object_name}"
        self.pipeline.run_module(mod, self.workspace)
        keys = self._get_feature_keys(f"{object_name}", mod)
        self.keys[f"{MEASUREMENT_CLASS_LABEL}"] = keys
        self._update_results(f"{MEASUREMENT_CLASS_LABEL}", self._get_results(
            f"{object_name}",
            keys
        ))
        return self.results[f"{MEASUREMENT_CLASS_LABEL}"]
