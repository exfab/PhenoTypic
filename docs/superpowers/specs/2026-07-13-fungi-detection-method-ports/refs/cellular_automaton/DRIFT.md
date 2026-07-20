# A06 TrickTrack CA drift register

Each row is one intentional difference from the pinned C++ source. There are no unregistered
semantic deviations in the approved core.

| ID | Deviation | Reason and consequence | Evidence |
|---|---|---|---|
| D01 | Cell objects and `std::vector` neighbors become three one-dimensional int64 CSR arrays. | This import-cheap numerical boundary avoids detector objects. CSR cell order, neighbor insertion order, and duplicate entries are preserved exactly. | `_cellular_automaton.py:40-70`; source `CMCell.h:192-195,227-229`; source-fixture test |
| D02 | Python validates ndarray container, int64 dtype, one-dimensional shape, CSR structure, and in-range indices. | C++ relies on typed containers and valid caller indices. Rejection prevents memory-unsafe or ambiguous inputs before evolution. Valid inputs are unchanged. | `_cellular_automaton.py:29-70`; `test_invalid_inputs_raise` |
| D03 | Python rejects `bool` and constrains `3 <= min_hits_per_track <= 257`. | The lower bound prevents C++ unsigned subtraction underflow in `min_hits - 2`; the upper bound is the largest request whose required source `unsigned char` state fits at 255. | `_cellular_automaton.py:72-79`; source `HitChainMaker.ipp:86-88,111`; upper/lower fixture cases |
| D04 | Empty cell graphs and empty root lists have deterministic empty array results. | The source has no explicit public empty-result contract. The Python result remains shape- and dtype-stable without inventing cells or roots. | `_cellular_automaton.py:224-246`; `test_empty_graph_and_isolated_root_have_deterministic_empty_results` |
| D05 | Ordered C++ path vectors become int64 `path_offsets` plus flattened `path_cell_indices`. | This canonical ragged-array representation retains path order, cell order, duplicates, cycles, and exact length without returning Python object arrays. | `_cellular_automaton.py:155-183,237-245`; source `CMCell.h:203-220`; fixture all-path comparison |
| D06 | Root cells are supplied directly rather than derived from TrickTrack's detector-layer graph. | The approved core excludes detector graph construction. Supplied order is load-bearing and is preserved. | `_cellular_automaton.py:186-239`; source `HitChainMaker.ipp:104-117`; root-order counterexample |
| D07 | Public outputs own their arrays and do not alias caller inputs. | This follows the PhenoTypic numerical-helper immutability rule. Evolution never mutates CSR inputs, so source behavior on valid graphs is unchanged. | `_cellular_automaton.py:209-212,224-246`; `test_inputs_are_not_mutated_or_aliased_by_outputs` |
