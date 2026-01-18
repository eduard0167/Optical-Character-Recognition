#include "randomForest.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <set>
#include "decisionTree.h"

using std::vector;
using std::pair;
using std::string;
using std::mt19937;

vector<vector<int>> get_random_samples(const vector<vector<int>> &samples,
                                       int num_to_return) {
    // Returns a vector of size num_to_return with random, unique elements
    // from samples
    vector<vector<int>> ret;

    int size = samples.size();
    std::random_device rd;
    mt19937 mt(rd());
    std::set<int> rows;

    while (rows.size() != num_to_return) {
        rows.insert(mt() % size);
    }

    for (std::set<int>::iterator it = rows.begin(); it != rows.end(); ++it) {
        ret.push_back(samples[*it]);
    }

    return ret;
}

RandomForest::RandomForest(int num_trees, const vector<vector<int>> &samples)
    : num_trees(num_trees), images(samples) {}

void RandomForest::build() {
    // Allocates n / num_trees for each Tree
    // Where n is the total number of training tests
    // Then trains each tree with the chosen tests
    assert(!images.empty());
    vector<vector<int>> random_samples;

    int data_size = images.size() / num_trees;

    for (int i = 0; i < num_trees; i++) {
        // cout << "Creating Tree no: " << i << endl;
        random_samples = get_random_samples(images, data_size);

        // Builds a new Tree and trains it
        trees.push_back(Node());
        trees[trees.size() - 1].train(random_samples);
    }
}

int RandomForest::predict(const vector<int> &image) {
    // Returns the most probable prediction for the test in the argument
    // each Tree will be queried and the final answer will be considered
    // the majority one
    int freq[10] = {0};
    for (int i = 0; i < num_trees; i++) {
        freq[trees[i].predict(image)]++;
    }

    int result = 0;
    int max_freq = freq[0];
    for (int i = 1; i < num_trees; i++) {
        if (max_freq < freq[i]) {
            max_freq = freq[i];
            result = i;
        }
    }
    
    return result;
}
