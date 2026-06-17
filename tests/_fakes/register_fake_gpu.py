"""Import side effect: register ``FakeGpuDetector`` into the phenotypic namespace.

Used via ``PHENOTYPIC_PRELOAD_MODULES=tests._fakes.register_fake_gpu`` so a fresh
SLURM worker process can deserialize a pipeline containing the test-only
``FakeGpuDetector`` (``ImagePipeline.from_json`` resolves op classes from the
``phenotypic`` namespace). In-process unit tests use a ``monkeypatch`` fixture
instead; this module exists only for the live SLURM dispatch test, where the
stage workers run in separate processes that the fixture cannot reach.
"""

import phenotypic

from tests._fakes.fake_gpu_detector import FakeGpuDetector

phenotypic.FakeGpuDetector = FakeGpuDetector
