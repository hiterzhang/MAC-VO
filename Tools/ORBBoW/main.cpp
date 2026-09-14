#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <openssl/sha.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "include/ORBVocabulary.h"

namespace {

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) fields.push_back(field);
    return fields;
}

std::string sha256_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to read vocabulary");
    SHA256_CTX context;
    SHA256_Init(&context);
    std::vector<char> buffer(1 << 20);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) SHA256_Update(&context, buffer.data(), count);
    }
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256_Final(digest, &context);
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (unsigned char byte : digest) result << std::setw(2) << static_cast<int>(byte);
    return result.str();
}

std::vector<cv::Mat> rows(const cv::Mat& descriptors) {
    std::vector<cv::Mat> result;
    result.reserve(descriptors.rows);
    for (int index = 0; index < descriptors.rows; ++index) {
        result.push_back(descriptors.row(index));
    }
    return result;
}

struct Candidate {
    long frame_id;
    double score;
};

}  // namespace

int main(int argc, char** argv) {
    std::string vocabulary_path;
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == "--vocabulary") {
            vocabulary_path = argv[index + 1];
            ++index;
        }
    }
    if (vocabulary_path.empty()) {
        std::cerr << "--vocabulary is required\n";
        return 2;
    }

    ORB_SLAM3::ORBVocabulary vocabulary;
    if (!vocabulary.loadFromTextFile(vocabulary_path)) {
        std::cerr << "failed to load vocabulary\n";
        return 3;
    }
    auto orb = cv::ORB::create(1200);
    std::unordered_map<long, DBoW2::BowVector> database;
    std::cout << "READY\t1\t" << sha256_file(vocabulary_path)
              << "\t" << CV_VERSION << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "STOP") {
            std::cout << "BYE" << std::endl;
            return 0;
        }
        if (line == "PING") {
            std::cout << "PONG" << std::endl;
            continue;
        }
        try {
            const auto fields = split_tabs(line);
            if (fields.size() != 6 || fields[0] != "QUERY") {
                throw std::runtime_error("expected QUERY with five arguments");
            }
            const long request_id = std::stol(fields[1]);
            const long target_id = std::stol(fields[2]);
            const std::string image_path = fields[3];
            const long min_gap = std::stol(fields[4]);
            const int top_k = std::stoi(fields[5]);
            if (request_id < 0 || target_id < 0 || min_gap < 0 || top_k < 1) {
                throw std::runtime_error("numeric arguments are invalid");
            }
            cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
            if (image.empty()) throw std::runtime_error("unable to read image");
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
            DBoW2::BowVector current;
            if (!descriptors.empty()) vocabulary.transform(rows(descriptors), current);

            std::vector<Candidate> candidates;
            for (const auto& item : database) {
                if (target_id - item.first < min_gap) continue;
                candidates.push_back({item.first, vocabulary.score(current, item.second)});
            }
            std::sort(candidates.begin(), candidates.end(), [](const Candidate& left, const Candidate& right) {
                if (left.score != right.score) return left.score > right.score;
                return left.frame_id < right.frame_id;
            });
            if (static_cast<int>(candidates.size()) > top_k) candidates.resize(top_k);
            database[target_id] = current;

            std::cout << "RESULT\t" << request_id << "\t" << target_id
                      << "\t" << candidates.size();
            std::cout << std::setprecision(17);
            for (const auto& candidate : candidates) {
                std::cout << "\t" << candidate.frame_id << ":" << candidate.score;
            }
            std::cout << std::endl;
        } catch (const std::exception& error) {
            std::cout << "ERROR\t" << error.what() << std::endl;
        }
    }
    return 0;
}
