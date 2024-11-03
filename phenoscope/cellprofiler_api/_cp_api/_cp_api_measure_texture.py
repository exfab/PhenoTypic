from cellprofiler.modules.measuretexture import MeasureTexture
from numpy import iterable

import os
import logging
formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

# ----- Pkg Relative Import -----
from ._cp_api_measure_intensity import CellProfilerApiMeasureIntensity

# ----- Global Constants -----
ORIGIN_METHOD_LABEL = "origin_method"
METRIC_LABEL = "Metric"


# ----- Main Class Definition -----
# TODO: Cleanup & optimize setting calls
class CellProfilerApiMeasureTexture(CellProfilerApiMeasureIntensity):
    def measure_texture(self, object_name=None, image_name=None,texture_scale: iterable = None, gray_levels: int = 256):
        try:
            return self._measure_texture(object_name=object_name, image_name=image_name,texture_scale=texture_scale, gray_levels=gray_levels)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            log.warning(f"could not measure texture for {self.image_name}", exc_info=True)
            self.status_validity = False
            return None

    def _measure_texture(self, object_name=None, image_name=None, texture_scale: iterable = None, gray_levels: int = 256):
        """
        :param scale:
        :param gray_levels:
        :return:
        --- Description ---
        Measurements are computed in 4 different directions:
            00 = Horizontal
            01 = Diagonal (135 deg)
            02 = Vertical (90 deg)
            04 = Diagonal (45 deg)
        NOTE:
            Metric Metadata Interpretation = {Class}_{Feature}_{img_name}_{scale}_{dir}_{gray_lvl}

        """
        MEASUREMENT_CLASS_LABEL = "Texture"
        if object_name is None:
            object_name = self.colony_name
        if image_name is None:
            image_name=self.image_name
        if texture_scale is None:
            scale = [5]
        else:
            scale=texture_scale
        metric_axis_map = {  # Measurement Axis Translation
            "00": "deg(0)",  # -
            "01": "deg(135)",  # \
            "02": "deg(90)",  # -
            "03": "deg(45)"  # /
        }
        mod = MeasureTexture()
        mod.gray_levels.value = gray_levels
        mod.images_or_objects.value = "Objects"
        mod.images_list.value = image_name
        mod.objects_list.value = object_name
        # May potentially integrate image or object switch in future versions
        for s in scale:
            mod.add_scale(s)

        self.pipeline.run_module(mod, self.workspace)

        keys = self._get_feature_keys(object_name, mod)
        self.keys[f"{MEASUREMENT_CLASS_LABEL}"] = keys

        results = self._get_results(
            object_name, keys
        ).reset_index(drop=False)
        metadata = results["Metric"].str.replace(image_name, "source")

        metadata = metadata.str.split("_", expand=True).rename(
            columns={
                0: "origin_method",
                1: "feature",
                2: "source",
                3: "scale",
                4: "axis",
                5: "gray"
            }

        )

        metadata["gray"] = metadata["gray"].apply(lambda x: f"gray({x})")
        metadata["scale"] = metadata["scale"].apply(lambda x: f"scale({x})")
        metadata['axis'] = metadata['axis'].apply(lambda x: metric_axis_map[x])
        metadata = metadata["origin_method"] + "_" \
                   + metadata["feature"] + "_" \
                   + metadata["scale"] + "_" \
                   + metadata["axis"] + "_" \
                   + metadata["gray"]
        results["Metric"] = metadata
        self._update_results(f"{MEASUREMENT_CLASS_LABEL}", results.set_index("Metric"))
        return self.results[MEASUREMENT_CLASS_LABEL]
