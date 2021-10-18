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
    // TODO(you)
    // Intoarce un vector de marime num_to_return cu elemente random,
    // diferite din samples
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
    // Aloca pentru fiecare Tree cate n / num_trees
    // Unde n e numarul total de teste de training
    // Apoi antreneaza fiecare tree cu testele alese
    assert(!images.empty());
    vector<vector<int>> random_samples;

    int data_size = images.size() / num_trees;

    for (int i = 0; i < num_trees; i++) {
        // cout << "Creating Tree nr: " << i << endl;
        random_samples = get_random_samples(images, data_size);

        // Construieste un Tree nou si il antreneaza
        trees.push_back(Node());
        trees[trees.size() - 1].train(random_samples);
    }
}

int RandomForest::predict(const vector<int> &image) {
    // TODO(you)
    // Va intoarce cea mai probabila prezicere pentru testul din argument
    // se va interoga fiecare Tree si se va considera raspunsul final ca
    // fiind cel majoritar
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
