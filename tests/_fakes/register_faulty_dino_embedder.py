"""Import side effect: register ``FaultyDinoEmbedding`` into the phenotypic namespace.

For ``PHENOTYPIC_PRELOAD_MODULES``, so a fresh process (a CLI subprocess, a
spawned Stage-4 DataLoader worker) can deserialize a pipeline holding it.
"""

import phenotypic

from tests._fakes.faulty_dino_embedder import FaultyDinoEmbedding

phenotypic.FaultyDinoEmbedding = FaultyDinoEmbedding
