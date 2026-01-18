#include "./decisionTree.h"  // NOLINT(build/include)
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <set>
#include <map>
#include <random>

using std::string;
using std::pair;
using std::vector;
using std::unordered_map;
using std::make_shared;
using std::mt19937;

// Decision tree node structure
// splitIndex = the dimension used for splitting
// split_value = the value used for splitting
// is_leaf and result are used when we have a leaf node
Node::Node() {
    is_leaf = false;
    left = nullptr;
    right = nullptr;
}

void Node::make_decision_node(const int index, const int val) {
    split_index = index;
    split_value = val;
}

void Node::make_leaf(const vector<vector<int>> &samples,
                     const bool is_single_class) {
    // Sets the node as a leaf (modifies is_leaf and result)
    // is_single_class = true -> all tests have the same class (that is the result)
    // is_single_class = false -> chooses the most frequent class
    is_leaf = true;

    if (is_single_class) {
        result = samples[0][0];
    } else {
        int freq[10] = {0};
        int size = samples.size();

        for (int i = 0; i < size; i++) {
            freq[samples[i][0]]++;
        }

        int max_freq = freq[0], value = 0;;
        for (int i = 1; i < size; i++) {
            if (max_freq < freq[i]) {
                value = i;
                max_freq = freq[i];
            }
        }

        result = value;
    }
}

pair<int, int> find_best_split(const vector<vector<int>> &samples,
                               const vector<int> &dimensions) {
    // Returns the best split dimension and value from the given tests.
    // By best split (dimension and value) we mean the split that maximizes IG.
    // The returned pair consists of (split_index, split_value).
    int splitIndex = -1, splitValue = -1;
    int size = samples.size();
    int nr_col = dimensions.size();
    float maxIG = -1;

    float parentEntropy = get_entropy(samples);

    for (int i = 0; i < nr_col; i++) {
        vector<int> test_values = compute_unique(samples, dimensions[i]);
        // nr_values = number of values for which IG will be calculated
        int nr_values = test_values.size();

        for (int j = 0; j < nr_values; j++) {
            pair<vector<int>, vector<int>> split_indexes = get_split_as_indexes
                                    (samples, dimensions[i], test_values[j]);

            // If the split does not form a left or right child (not a good
            // split), it is skipped.
            if (split_indexes.first.size() == 0 ||
                                split_indexes.second.size() == 0) {
                continue;
            }

            // Calculate Information Gain for the current split.
            float IG = parentEntropy - (split_indexes.first.size() *
                        get_entropy_by_indexes(samples, split_indexes.first) +
                        split_indexes.second.size() *
                        get_entropy_by_indexes(samples, split_indexes.second)) /
                        size;

            if (IG > maxIG) {
                maxIG = IG;
                splitIndex = dimensions[i];
                splitValue = test_values[j];
            }
        }
    }

    return pair<int, int>(splitIndex, splitValue);
}

void Node::train(const vector<vector<int>> &samples) {
    // Trains the current node and its children, if necessary.
    // 1) Checks if all received tests have the same class (result).
    // If yes, this node becomes a leaf, otherwise continues the algorithm.
    // 2) If no valid split exists, this node becomes a leaf. Otherwise,
    // takes the best split and recursively continues.
    int size = samples.size();

    if (same_class(samples)) {
        make_leaf(samples, true);
    } else {
        // best_split = index and value of the best split. The vector of
        // dimensions formed to calculate best_split contains random elements
        // between 0 and the number of columns in samples (785 for us).
        pair<int, int> best_split = find_best_split(samples,
                                        random_dimensions(samples[0].size()));

        // If no valid split exists, the node becomes a leaf.
        if (best_split.first == -1 || best_split.second == -1) {
            make_leaf(samples, false);
        } else {
            // If a valid split exists, the current node becomes a decision node.
            make_decision_node(best_split.first, best_split.second);

            // Calculate the two subsets obtained after splitting.
            pair<vector<vector<int>>, vector<vector<int>>> split_sets =
                                    split(samples, split_index, split_value);
            // Tree formation continues recursively with the two subtrees.
            left = make_shared<Node> (Node());
            left->train(split_sets.first);

            right = make_shared<Node> (Node());
            right->train(split_sets.second);
        }
    }
}


int Node::predict(const vector<int> &image) const {
    // Returns the result predicted by the decision tree
    if (is_leaf) {
        return result;
    }

    if (image[split_index - 1] <= split_value) {
        return left->predict(image);
    } else {
        return right->predict(image);
    }
}

bool same_class(const vector<vector<int>> &samples) {
    // Checks if the tests received as argument all have the same
    // class (result). It is used in train to determine if
    // it makes sense to search for splits anymore.
    int num = samples[0][0];
    int size = samples.size();

    for (int i = 1; i < size; i++) {
        if (samples[i][0] != num) {
            return false;
        }
    }

    return true;
}

float get_entropy(const vector<vector<int>> &samples) {
    // Returns the entropy of the received tests
    assert(!samples.empty());
    vector<int> indexes;

    int size = samples.size();
    for (int i = 0; i < size; i++) {
        indexes.push_back(i);
    }

    return get_entropy_by_indexes(samples, indexes);
}

float get_entropy_by_indexes(const vector<vector<int>> &samples,
                             const vector<int> &index) {
    // Returns the entropy of the subset from the total test set (samples).
    // Provided that the subset contains tests whose indices are found in
    // the index vector (Only lines in the index vector are considered).

    vector<float> freq(10, 0.0f);
    int size = index.size();
    float entropy = 0.0f;

    // Calculate the number of occurrences of each digit, then calculate
    // entropy.
    for (int i = 0; i < size; i++) {
        freq[samples[index[i]][0]]++;
    }

    for (int i = 0; i < 10; i++) {
        freq[i] /= size;

        if (freq[i] > 0.0f)
            entropy += freq[i] * log2(freq[i]);
    }

    return -entropy;
}

vector<int> compute_unique(const vector<vector<int>> &samples, const int col) {
    // Returns all unique values (duplicates are removed)
    // that appear in the test set, on column col
    vector<int> uniqueValues;

    std::set<int> values;
    int size = samples.size();

    for (int i = 0; i < size; i++) {
        values.insert(samples[i][col]);
    }

    for (std::set<int>::iterator it = values.begin(); it != values.end(); ++it) {
        uniqueValues.push_back(*it);
    }

    return uniqueValues;
}

pair<vector<vector<int>>, vector<vector<int>>> split(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // Returns the 2 subsets of tests obtained after separation
    // Based on split_index and split_value
    vector<vector<int>> left, right;

    auto p = get_split_as_indexes(samples, split_index, split_value);
    for (const auto &i : p.first) left.push_back(samples[i]);
    for (const auto &i : p.second) right.push_back(samples[i]);

    return pair<vector<vector<int>>, vector<vector<int>>>(left, right);
}

pair<vector<int>, vector<int>> get_split_as_indexes(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // Returns the indices of samples from the 2 subsets obtained after
    // separation based on split_index and split_value
    vector<int> left, right;

    for (int i = 0; i < samples.size(); i++) {
        if (samples[i][split_index] <= split_value) {
            left.push_back(i);
        } else {
            right.push_back(i);
        }
    }

    return make_pair(left, right);
}

vector<int> random_dimensions(const int size) {
    // Returns sqrt(size) different dimensions to search for the max split.
    // Note: Found dimensions are > 0 and < size
    const int squareRoot = floor(sqrt(size));
    vector<int> rez;
    rez.reserve(squareRoot);

    std::set<int> values;
    std::random_device rd;
    mt19937 mt(rd());

    while (values.size() != squareRoot) {
        int res = mt() % size;
        if (res > 0) {
            values.insert(res);
        }
    }

    for (std::set<int>::iterator it = values.begin(); it != values.end(); ++it) {
        rez.push_back(*it);
    }

    return rez;
}
