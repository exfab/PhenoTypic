#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vaa3d/app2/fastmarching_dt.h"

namespace {

const unsigned char kStandardImage[] = {
    0, 0, 0, 0, 0,
    0, 2, 8, 3, 0,
    0, 4, 1, 7, 0,
    0, 6, 5, 9, 0,
    0, 0, 0, 0, 0,
};

const unsigned char kDiagonalImage[] = {
    0, 100,
    100, 1,
};

const unsigned char kThresholdImage[] = {1, 2, 5};
const unsigned char kAllBackgroundImage[] = {1, 2};
const unsigned char kNoBackgroundImage[] = {1, 2};
const unsigned char kPostFrontierDiagonalImage[] = {
    0, 100, 100,
    100, 1, 100,
    100, 100, 1,
};

void write_array(
    std::ofstream& output,
    const char* label,
    const float* values,
    int size
) {
    output << label;
    output << std::setprecision(9);
    for (int index = 0; index < size; ++index) {
        output << ' ' << values[index];
    }
    output << '\n';
}

void write_reference_cost(std::ofstream& output, const float* distance, int size) {
    const auto bounds = std::minmax_element(distance, distance + size);
    const double minimum = *bounds.first;
    const double span = *bounds.second - minimum;
    output << "COST";
    output << std::setprecision(9);
    for (int index = 0; index < size; ++index) {
        const int lookup_index = static_cast<int>(
            (static_cast<double>(distance[index]) - minimum) / span * 255.0
        );
        output << ' ' << givals[lookup_index];
    }
    output << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr
            << "usage: source_harness <case> <cnn_type:1|2> <output.txt>\n";
        return 2;
    }

    const std::string case_name(argv[1]);
    const int cnn_type = std::stoi(argv[2]);
    if (cnn_type != 1 && cnn_type != 2) {
        std::cerr << "cnn_type must be 1 or 2\n";
        return 2;
    }

    int rows = 0;
    int columns = 0;
    int background_threshold = 0;
    std::vector<unsigned char> image;
    if (case_name == "standard") {
        rows = 5;
        columns = 5;
        image.assign(std::begin(kStandardImage), std::end(kStandardImage));
    } else if (case_name == "diagonal") {
        rows = 2;
        columns = 2;
        image.assign(std::begin(kDiagonalImage), std::end(kDiagonalImage));
    } else if (case_name == "threshold") {
        rows = 1;
        columns = 3;
        background_threshold = 2;
        image.assign(std::begin(kThresholdImage), std::end(kThresholdImage));
    } else if (case_name == "all_background") {
        rows = 1;
        columns = 2;
        background_threshold = 2;
        image.assign(
            std::begin(kAllBackgroundImage), std::end(kAllBackgroundImage)
        );
    } else if (case_name == "no_background") {
        rows = 1;
        columns = 2;
        image.assign(std::begin(kNoBackgroundImage), std::end(kNoBackgroundImage));
    } else if (case_name == "post_frontier_diagonal") {
        rows = 3;
        columns = 3;
        image.assign(
            std::begin(kPostFrontierDiagonalImage),
            std::end(kPostFrontierDiagonalImage)
        );
    } else {
        std::cerr << "unknown source case\n";
        return 2;
    }
    float* distance = nullptr;
    if (!fastmarching_dt(
            image.data(), distance, columns, rows, 1, cnn_type, background_threshold
        )) {
        std::cerr << "fastmarching_dt failed\n";
        return 1;
    }

    std::ofstream output(argv[3]);
    if (!output) {
        delete[] distance;
        throw std::runtime_error("could not open output file");
    }
    write_array(output, "DISTANCE", distance, rows * columns);
    const bool finite_span = *std::max_element(distance, distance + rows * columns)
        > *std::min_element(distance, distance + rows * columns);
    if (finite_span) {
        write_reference_cost(output, distance, rows * columns);
    }
    delete[] distance;
    return 0;
}
