# A10 FilFinder reference attribution

The local `upstream/` corpus was extracted from the official `fil-finder==1.8` source
distribution. FilFinder is Copyright 2014-2020 Eric Koch and Erik Rosolowsky and licensed under
the MIT License. The complete license text is retained at `upstream/LICENSE.rst:1-21`; package
version, dependency, and license metadata are retained at `upstream/PKG-INFO:1-29`.

The paper is Eric W. Koch and Erik W. Rosolowsky, "Filament Identification through Mathematical
Morphology," *Monthly Notices of the Royal Astronomical Society* 452(4), 3435-3450 (2015),
doi:10.1093/mnras/stv1521. The committed PDF is the authors' arXiv preprint 1507.02289. The paper
is more than five years old; it is used for algorithm context, while the maintained 1.8 source is
the executable authority for wrapper behavior.

The reference source, paper, generator, and fixture are audit material. They are not imported by
PhenoTypic at runtime and remain outside built wheel and sdist artifacts. The production wrapper
uses the separately installed MIT-licensed `fil-finder` dependency through the `topology` extra.
The root `NOTICE` update remains integrator-owned.
