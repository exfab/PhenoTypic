Glossary
========

.. glossary::
   :sorted:

   colony
      A visible microbial growth on an agar plate surface. In PhenoTypic,
      a colony corresponds to a connected component in the object map.

   detect_mat
      The detection matrix — an enhanced grayscale representation of the
      plate image that detectors consume. Enhancers modify ``detect_mat``
      while leaving ``rgb`` and ``gray`` unchanged.

   grid section
      A rectangular region of a :term:`GridImage` corresponding to one
      well in the physical plate. Indexed by (row, column) or flattened
      section number.

   GridImage
      An :term:`Image` subclass that adds grid awareness — row/column
      layout, section extraction, and grid-aware detection/refinement.
      Use for arrayed colony plates (96-well, 384-well).

   Image
      PhenoTypic's central data structure. Stores pixel data (rgb, gray),
      detection results (objmask, objmap), metadata, and provides accessor-
      based access to all components.

   ImagePipeline
      A serializable chain of operations (enhancers, detectors, refiners)
      and measurements. Apply to images with ``.apply()`` or
      ``.apply_and_measure()``. Save/load with ``.to_json()`` /
      ``.from_json()``.

   inoculum
      The initial microbial deposit on the agar surface. In filamentous
      fungi detection, the dense central region from which hyphae radiate.

   objmap
      The labeled object map — a uint16 array where each detected colony
      receives a unique integer label (0 = background). Written by
      detectors, refined by refiners.

   objmask
      The binary object mask — a boolean array where ``True`` indicates
      colony pixels and ``False`` indicates background. Written by
      detectors, refined by refiners.

   prefab pipeline
      A pre-configured :term:`ImagePipeline` subclass tuned for a specific
      organism or imaging scenario (e.g., ``HeavyOtsuPipeline``,
      ``FilamentousFungiPipeline``).

   well
      A physical position on an arrayed agar plate (e.g., one of 96
      positions in a 96-well format). Corresponds to a :term:`grid section`
      in PhenoTypic.

   accessor
      A property on :term:`Image` that provides NumPy-style indexing to
      image data. Primary accessors: ``rgb``, ``gray``, ``detect_mat``,
      ``objmask``, ``objmap``. High-level accessors: ``color``, ``grid``,
      ``metadata``, ``plot``.

   hyphae
      Branching filamentous structures produced by filamentous fungi.
      Singular: hypha. The network of hyphae is called mycelium.

   mycelium
      The network of branching :term:`hyphae` that forms the body of a
      filamentous fungus. Detected by ``FilamentousFungiDetector``.
