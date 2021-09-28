// copyright Luca Istrate, Andrei Medar

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

// structura unui nod din decision tree
// splitIndex = dimensiunea in functie de care se imparte
// split_value = valoarea in functie de care se imparte
// is_leaf si result sunt pentru cazul in care avem un nod frunza
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
    // TODO(you)
    // Seteaza nodul ca fiind de tip frunza (modificati is_leaf si result)
    // is_single_class = true -> toate testele au aceeasi clasa (acela e result)
    // is_single_class = false -> se alege clasa care apare cel mai des
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
    // Intoarce cea mai buna dimensiune si valoare de split dintre testele
    // primite. Prin cel mai bun split (dimensiune si valoare)
    // ne referim la split-ul care maximizeaza IG
    // pair-ul intors este format din (split_index, split_value)
    int splitIndex = -1, splitValue = -1;
    int size = samples.size();
    int nr_col = dimensions.size();
    float maxIG = -1;

    float parentEntropy = get_entropy(samples);

    for (int i = 0; i < nr_col; i++) {
        vector<int> test_values = compute_unique(samples, dimensions[i]);
        // nr_values = numarul de valori pentru care se va calcula IG
        int nr_values = test_values.size();

        for (int j = 0; j < nr_values; j++) {
            pair<vector<int>, vector<int>> split_indexes = get_split_as_indexes
                                    (samples, dimensions[i], test_values[j]);

            // Daca split-ul nu formeaza copil stang sau drept (nu este split
            // bun) se renunta la acest split.
            if (split_indexes.first.size() == 0 ||
                                split_indexes.second.size() == 0) {
                continue;
            }

            // Se calculeaza Information Gain pentru split-ul curent.
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
    // Antreneaza nodul curent si copii sai, daca e nevoie
    // 1) verifica daca toate testele primite au aceeasi clasa (raspuns)
    // Daca da, acest nod devine frunza, altfel continua algoritmul.
    // 2) Daca nu exista niciun split valid, acest nod devine frunza. Altfel,
    // ia cel mai bun split si continua recursiv
    int size = samples.size();

    if (same_class(samples)) {
        make_leaf(samples, true);
    } else {
        // best_split = indexul si valoarea celui mai bun split. Vectorul de
        // dimensiuni format pentru calcularea lui best_split contine elemente
        // random intre 0 si numarul de coloane din samples (785 pentru noi).
        pair<int, int> best_split = find_best_split(samples,
                                        random_dimensions(samples[0].size()));

        // Daca nu exista niciun split valid, nodul devine frunza.
        if (best_split.first == -1 || best_split.second == -1) {
            make_leaf(samples, false);
        } else {
            // Daca exista un split valid, nodul curent devine nod de decizie.
            make_decision_node(best_split.first, best_split.second);

            // Se calculeaza cele doua subseturi obtinute in urma split-ului.
            pair<vector<vector<int>>, vector<vector<int>>> split_sets =
                                    split(samples, split_index, split_value);
            // Formarea arborelui continua recursiv cu cei doi subarbori.
            left = make_shared<Node> (Node());
            left->train(split_sets.first);

            right = make_shared<Node> (Node());
            right->train(split_sets.second);
        }
    }
}


int Node::predict(const vector<int> &image) const {
    // TODO(you)
    // Intoarce rezultatul prezis de catre decision tree
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
    // TODO(you)
    // Verifica daca testele primite ca argument au toate aceeasi
    // clasa(rezultat). Este folosit in train pentru a determina daca
    // mai are rost sa caute split-uri
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
    // Intoarce entropia testelor primite
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
    // TODO(you)
    // Intoarce entropia subsetului din setul de teste total(samples)
    // Cu conditia ca subsetul sa contina testele ale caror indecsi se gasesc in
    // vectorul index (Se considera doar liniile din vectorul index)

    vector<float> freq(10, 0.0f);
    int size = index.size();
    float entropy = 0.0f;

    // Se calculeaza numarul de aparitii al fiecarei cifre, apoi se calculeaza
    // entropia.
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
    // TODO(you)
    // Intoarce toate valorile (se elimina duplicatele)
    // care apar in setul de teste, pe coloana col
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
    // Intoarce cele 2 subseturi de teste obtinute in urma separarii
    // In functie de split_index si split_value
    vector<vector<int>> left, right;

    auto p = get_split_as_indexes(samples, split_index, split_value);
    for (const auto &i : p.first) left.push_back(samples[i]);
    for (const auto &i : p.second) right.push_back(samples[i]);

    return pair<vector<vector<int>>, vector<vector<int>>>(left, right);
}

pair<vector<int>, vector<int>> get_split_as_indexes(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // TODO(you)
    // Intoarce indecsii sample-urilor din cele 2 subseturi obtinute in urma
    // separarii in functie de split_index si split_value
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
    // TODO(you)
    // Intoarce sqrt(size) dimensiuni diferite pe care sa caute splitul maxim
    // Precizare: Dimensiunile gasite sunt > 0 si < size
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
