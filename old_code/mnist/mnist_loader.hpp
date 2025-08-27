#pragma once

#include <fstream>
#include <iostream>
#include <string_view>

#include "tensor.hpp"
#include "utils.hpp"

namespace ml {

using MNISTDataType = Tensor<float>;
using MNISTDataLabelType = Tensor<uint8_t>;

struct MNISTDataResult {
    MNISTDataType data;
    MNISTDataLabelType labels;
    bool success;
};

class MNISTLoader {
    static constexpr std::string_view TRAINING_SET_FILE = "data/train-images-idx3-ubyte";
    static constexpr std::string_view LABELS_FILE = "data/train-labels-idx1-ubyte";

    static constexpr std::string_view TEST_SET_FILE = "data/t10k-images-idx3-ubyte";
    static constexpr std::string_view TEST_LABELS_FILE = "data/t10k-labels-idx1-ubyte";

   public:
    MNISTLoader() = default;

    MNISTDataResult load_training_data() { return load_data(TRAINING_SET_FILE, LABELS_FILE); }

    MNISTDataResult load_testing_data() { return load_data(TEST_SET_FILE, TEST_LABELS_FILE); }

   private:
    MNISTDataResult load_data(std::string_view data_file_path, std::string_view label_file_path) {
        MNISTDataType data = parse_data(data_file_path);
        MNISTDataLabelType labels = parse_labels(label_file_path);

        const bool success = labels.size() > 0 && data.size() > 0;

        return {std::move(data), std::move(labels), success};
    }

    MNISTDataLabelType parse_labels(std::string_view label_file_path) {
        std::ifstream label_file(label_file_path, std::ios_base::in);

        if (!label_file.is_open()) {
            std::cerr << "Failed to open file: " << label_file_path << std::endl;
            return {};
        }

        int32_t magic_number = 0;
        int32_t num_images = 0;

        {
            label_file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = swap_bytes_32bit(magic_number);
            assertm(magic_number == 2049, "Invalid magic number for labels file");
        }

        {
            label_file.read((char*)&num_images, sizeof(num_images));
            num_images = swap_bytes_32bit(num_images);
            assertm(num_images > 0, "Invalid number of images");
        }

        MNISTDataLabelType labels{{static_cast<size_t>(num_images)}};

        for (int i = 0; i < num_images; i++) {
            uint8_t label = static_cast<uint8_t>(-1);
            label_file.read((char*)&label, sizeof(label));
            labels[i] = label;
        }

        return labels;
    }

    MNISTDataType parse_data(std::string_view data_file_path) {
        static constexpr size_t MAX_IMAGES = 10000;

        std::ifstream data_file(data_file_path, std::ios_base::in);

        if (!data_file.is_open()) {
            std::cerr << "Failed to open file: " << data_file_path << std::endl;
            return {};
        }

        size_t magic_number = 0;
        size_t num_images = 0;
        size_t num_rows = 0;
        size_t num_cols = 0;

        {
            int32_t magic_number_tmp = 0;
            data_file.read((char*)&magic_number_tmp, sizeof(magic_number_tmp));
            magic_number = static_cast<size_t>(swap_bytes_32bit(magic_number_tmp));
            assertm(magic_number == 2051, "Invalid magic number for data file");
        }

        {
            int32_t num_images_tmp = 0;
            data_file.read((char*)&num_images_tmp, sizeof(num_images_tmp));
            num_images =
                std::min(static_cast<size_t>(swap_bytes_32bit(num_images_tmp)), MAX_IMAGES);
            assertm(num_images > 0, "Invalid number of images");
        }

        {
            int32_t num_rows_tmp = 0;
            data_file.read((char*)&num_rows_tmp, sizeof(num_rows_tmp));
            num_rows = static_cast<size_t>(swap_bytes_32bit(num_rows_tmp));
            assertm(num_rows > 0, "Invalid number of rows");
        }

        {
            int32_t num_cols_tmp = 0;
            data_file.read((char*)&num_cols_tmp, sizeof(num_cols_tmp));
            num_cols = static_cast<size_t>(swap_bytes_32bit(num_cols_tmp));
            assertm(num_cols > 0, "Invalid number of cols");
        }

        MNISTDataType data{{num_images, num_rows, num_cols}};

        for (size_t i = 0; i < num_images; i++) {
            uint8_t pixel = static_cast<uint8_t>(-1);

            for (size_t r = 0; r < num_rows; r++) {
                for (size_t c = 0; c < num_cols; c++) {
                    data_file.read((char*)&pixel, sizeof(pixel));
                    // Scale pixel values to [0, 1]
                    data[{i, r, c}] = static_cast<float>(pixel) / 255.f;
                }
            }

            if (i % 1000 == 0) {
                std::cout << "Loaded " << i << " images" << std::endl;
            }
        }
        return data;
    }
};

}  // namespace ml
