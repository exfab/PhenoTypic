from __future__ import annotations

from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Iterator
from pathlib import Path

from ._pipe_grid_search_base import PipelineGridSearchBase

if TYPE_CHECKING:
    from phenotypic.tools_.typing_ import GridSearchSaveData, GridSearchConfig
    from phenotypic import Image


class PipeGridSearchSubmitit(PipelineGridSearchBase):

    def submitit(self, image: Image, slurm_config: Dict):
        try:
            import submitit
        except ImportError:
            raise ImportError("Missing dependency: submitit. "
                              "Install the slurm optional dependency for phenotypic.")

        # Save original image to output dir, and create config paths
        self._prep_image(image=image)

        submitit_dir = self.output_dir / "submitit"
        submitit_dir.mkdir(exist_ok=True)

        executor = submitit.AutoExecutor(folder=submitit_dir)
        executor.update_parameters(**slurm_config)

        # Collect parent pipeline directories, before parameter sweep
        parent_pipe_dirpaths = [
            p for p in self.data_dir.iterdir()
            if p.is_dir()
        ]
        for pipe_cfg_dir in parent_pipe_dirpaths:
            # Collect individual pipeline folder paths
            pipe_cfgs_dirs = [
                p for p in pipe_cfg_dir.iterdir()
                if p.is_dir()
            ]
            n = len(pipe_cfgs_dirs)
            executor.map_array(
                    PipeGridSearchSubmitit._process_single_pipe_dir,
                    pipe_cfgs_dirs,
                    [self._image_pkl_path] * n,
                    [self.data2save] * n,

            )
