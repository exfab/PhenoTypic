#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "tricktrack/CMCell.h"
#include "tricktrack/HitDoublets.h"
#include "tricktrack/SpacePoint.h"

using Hit = tricktrack::SpacePoint<std::size_t>;
using Cell = tricktrack::CMCell<Hit>;
using Status = tricktrack::CMCellStatus;

struct Case {
  std::string name;
  std::vector<std::vector<unsigned int>> outer_neighbors;
  std::vector<unsigned int> roots;
  unsigned int min_hits;
};

template <typename Value>
void print_vector(const std::vector<Value>& values) {
  std::cout << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::cout << ",";
    std::cout << values[index];
  }
  std::cout << "]";
}

template <typename Value>
void print_nested_vector(const std::vector<std::vector<Value>>& values) {
  std::cout << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::cout << ",";
    print_vector(values[index]);
  }
  std::cout << "]";
}

std::vector<unsigned int> statuses(const std::vector<Status>& all_status) {
  std::vector<unsigned int> values;
  values.reserve(all_status.size());
  for (const auto& status : all_status) values.push_back(status.theCAState);
  return values;
}

std::vector<unsigned int> flags(const std::vector<Status>& all_status) {
  std::vector<unsigned int> values;
  values.reserve(all_status.size());
  for (const auto& status : all_status) values.push_back(status.hasSameStateNeighbors);
  return values;
}

std::vector<int> first_equal_neighbors(
    const Case& test_case,
    const std::vector<Status>& all_status,
    const std::vector<unsigned int>& cells) {
  std::vector<int> matches;
  matches.reserve(cells.size());
  for (const auto cell : cells) {
    int match = -1;
    const auto state = all_status[cell].theCAState;
    for (const auto neighbor : test_case.outer_neighbors[cell]) {
      if (all_status[neighbor].theCAState == state) {
        match = static_cast<int>(neighbor);
        break;
      }
    }
    matches.push_back(match);
  }
  return matches;
}

void print_case(const Case& test_case) {
  std::vector<Hit> inner_hits{Hit(1.0, 0.0, 0.0, 0)};
  std::vector<Hit> outer_hits{Hit(2.0, 0.0, 0.0, 0)};
  tricktrack::HitDoublets<Hit> doublets(inner_hits, outer_hits);
  for (std::size_t index = 0; index < test_case.outer_neighbors.size(); ++index) {
    doublets.add(0, 0);
  }

  std::vector<Cell> cells;
  cells.reserve(test_case.outer_neighbors.size());
  for (std::size_t index = 0; index < test_case.outer_neighbors.size(); ++index) {
    cells.emplace_back(&doublets, static_cast<int>(index), 0, 0);
  }
  for (std::size_t cell = 0; cell < test_case.outer_neighbors.size(); ++cell) {
    for (const auto neighbor : test_case.outer_neighbors[cell]) {
      cells[cell].tagAsOuterNeighbor(neighbor);
    }
  }

  std::vector<Status> all_status(cells.size());
  std::vector<unsigned int> every_cell(cells.size());
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    every_cell[cell] = static_cast<unsigned int>(cell);
  }
  const unsigned int ordinary_rounds = test_case.min_hits - 3;
  std::vector<std::vector<int>> ordinary_matches;
  std::vector<std::vector<unsigned int>> ordinary_flags;
  std::vector<std::vector<unsigned int>> ordinary_states;

  for (unsigned int round = 0; round < ordinary_rounds; ++round) {
    ordinary_matches.push_back(
        first_equal_neighbors(test_case, all_status, every_cell));
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
      cells[cell].evolve(static_cast<unsigned int>(cell), all_status);
    }
    ordinary_flags.push_back(flags(all_status));
    for (auto& status : all_status) status.updateState();
    ordinary_states.push_back(statuses(all_status));
  }

  std::vector<int> root_matches;
  std::vector<unsigned int> root_flags;
  std::vector<unsigned int> states_after_each_root;
  std::vector<unsigned int> retained_roots;
  for (const auto root : test_case.roots) {
    root_matches.push_back(
        first_equal_neighbors(test_case, all_status, {root})[0]);
    cells[root].evolve(root, all_status);
    root_flags.push_back(all_status[root].hasSameStateNeighbors);
    all_status[root].updateState();
    states_after_each_root.push_back(all_status[root].theCAState);
    if (all_status[root].isRootCell(test_case.min_hits - 2)) {
      retained_roots.push_back(root);
    }
  }

  std::vector<Cell::CMntuplet> paths;
  for (const auto root : retained_roots) {
    Cell::CMntuplet path{root};
    cells[root].findNtuplets(cells, paths, path, test_case.min_hits);
  }
  std::vector<unsigned int> path_offsets{0};
  std::vector<unsigned int> path_cells;
  for (const auto& path : paths) {
    path_cells.insert(path_cells.end(), path.begin(), path.end());
    path_offsets.push_back(static_cast<unsigned int>(path_cells.size()));
  }

  std::vector<unsigned int> csr_offsets{0};
  std::vector<unsigned int> csr_indices;
  for (const auto& neighbors : test_case.outer_neighbors) {
    csr_indices.insert(csr_indices.end(), neighbors.begin(), neighbors.end());
    csr_offsets.push_back(static_cast<unsigned int>(csr_indices.size()));
  }

  std::cout << "{\"name\":\"" << test_case.name << "\",";
  std::cout << "\"outer_neighbor_offsets\":";
  print_vector(csr_offsets);
  std::cout << ",\"outer_neighbor_indices\":";
  print_vector(csr_indices);
  std::cout << ",\"root_cell_indices\":";
  print_vector(test_case.roots);
  std::cout << ",\"min_hits_per_track\":" << test_case.min_hits;
  std::cout << ",\"ordinary_rounds\":" << ordinary_rounds;
  std::cout << ",\"ordinary_first_equal_neighbors\":";
  print_nested_vector(ordinary_matches);
  std::cout << ",\"ordinary_flags\":";
  print_nested_vector(ordinary_flags);
  std::cout << ",\"ordinary_states\":";
  print_nested_vector(ordinary_states);
  std::cout << ",\"root_first_equal_neighbors\":";
  print_vector(root_matches);
  std::cout << ",\"root_flags\":";
  print_vector(root_flags);
  std::cout << ",\"states_after_each_root\":";
  print_vector(states_after_each_root);
  std::cout << ",\"retained_root_indices\":";
  print_vector(retained_roots);
  std::cout << ",\"final_states\":";
  print_vector(statuses(all_status));
  std::cout << ",\"path_offsets\":";
  print_vector(path_offsets);
  std::cout << ",\"path_cell_indices\":";
  print_vector(path_cells);
  std::cout << "}";
}

Case upper_bound_case() {
  Case result{"upper-bound-chain", std::vector<std::vector<unsigned int>>(256), {0}, 257};
  for (unsigned int cell = 0; cell < 255; ++cell) {
    result.outer_neighbors[cell].push_back(cell + 1);
  }
  return result;
}

int main() {
  const std::vector<Case> cases{
      {"lower-bound", {{1}, {}}, {0}, 3},
      {"immediate-root-order", {{2}, {0}, {3}, {}}, {0, 1}, 4},
      {"ordered-fork",
       {{2, 1}, {5}, {4, 3}, {7}, {6}, {8}, {}, {}, {}},
       {0},
       5},
      {"cycle-exact-depth", {{1}, {0}}, {0}, 4},
      {"isolated", {{}}, {0}, 3},
      upper_bound_case(),
  };

  std::cout << "{\"source_commit\":\"b164fad1361505ff8dbf328107b645753ce331ac\",\"cases\":[";
  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (index != 0) std::cout << ",";
    print_case(cases[index]);
  }
  std::cout << "]}\n";
  return 0;
}
