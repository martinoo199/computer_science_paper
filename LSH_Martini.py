def experiment(buckets):

    import json
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import re

    with open('TVs_all_merged.json') as f:
        data_full = json.load(f)

    def bootstrap(data_dict :dict, n_samples):
        """Return a random sample with replacement of the input dictionary for bootstrapping purposes"""
        data = {}
        while len(data.keys()) < n_samples:
            random.choice(list(data_dict))
            key, value = random.choice(list(data_dict.items()))
            data[key] = value
        return data

    # Take a random sample with replacement of 60% of the original data
    data = bootstrap(data_full, 0.6*1664)

    # Extract titles, features, shops and brands from the dictionary.
    titles = []
    features = []
    shops = []
    keys = []
    for k, v in data.items():
        for j in range(len(data[k])):
            key = data[k][j]['modelID']
            feature = data[k][j]['featuresMap']
            title = data[k][j]['title'].lower()
            shop = data[k][j]['shop'].lower()
            keys.append(key)
            features.append(feature)
            titles.append(title)
            shops.append(shop)

    # Total number of duplicates in the dataset
    n_duplicates = len(keys) - len(set(keys))

    brands = list()
    for item in features: #those are all the brands
        if 'Brand' in item:
            brands.append(item['Brand'].lower())
        else:
            brands.append(0)
    set_brands = set(brands) #set of all brands

    # Generate one-hot encoded vectors of titles compared with set of model words

    product_number = 0
    real_pairs = []
    for k, v in data.items():
        real_pairs.append(list(range(product_number, product_number + len(v))))
        product_number +=  len(v)

    def process_title(title: str):
        """"Seperates the titles into model words using a regular expression
        """
        classifiers = re.findall(r'[a-zA-Z0-9]*(?:(?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[a-zA-Z0-9]*', title)
        return set([clean_model_value(model_value) for model_value in classifiers])

    def clean_model_value(string):
        string = string.replace('-inch', 'inch')
        string = string.replace(' inch', 'inch')
        string = string.replace('inches', 'inch')
        string = string.replace('- inches', 'inch')
        string = string.replace('-inches', 'inch')
        string = string.replace('diag', 'diagonal')
        string = string.replace('diag', 'diagonal')
        string = string.replace('diag.', 'diagonal')
        string = string.replace('diagonalonalonal', 'diagonal')
        string = string.replace('diagonalonal', 'diagonal')
        string = string.replace('\\', '')
        string = string.replace('\\,', '')
        string = string.replace('\\-', '')
        string = string.replace('(', '')
        string = string.replace(')', '')
        string = string.replace('-', '')
        return string

    def build_vocab(product_titles: list):
        """Generates a set of model words from the product titles. This can be used
        to construct the binary vectors for minhashing """
        full_set = {item for set_ in product_titles for item in set_}
        vocab = {}
        for i, title in enumerate(list(full_set)):
            vocab[title] = i
        return vocab

    def one_hot(model_words: list, vocab: dict):
        """One hot encode titles
        """
        vec = np.zeros(len(vocab))
        for title in model_words:
            idx = vocab[title]
            vec[idx] = 1
        return vec

    # build model word titles
    model_words = []
    for title in titles:
        model_words.append(process_title(title))

    # build vocab
    vocab = build_vocab(model_words)

    # one-hot encode our shingles
    titles_1hot = []
    for model_word_set in model_words:
        titles_1hot.append(one_hot(model_word_set, vocab))

    # stack into single numpy array
    titles_1hot = np.stack(titles_1hot)

    # Minhashing
    def minhash_arr(vocab: dict, resolution: int):
        """Generate a number of minhashes and minhash the one_hot vectors
        store in array size [resolution, length of vocab]"""
        length = len(vocab.keys())
        arr = np.zeros((resolution, length))
        for i in range(resolution):
            permutation = np.random.permutation(len(vocab)) + 1
            arr[i, :] = permutation.copy()
        return arr.astype(int)

    def get_signature(minhash, vector):
        # get index locations of every 1 value in vector
        idx = np.nonzero(vector)[0].tolist()
        # use index locations to pull only positive positions in minhash
        titles = minhash[:, idx]
        # find minimum value in each hash vector
        signature = np.min(titles, axis=1)
        return signature

    # Generate an array of minhashes and then the signatures
    num_hashes = 600
    arr = minhash_arr(vocab, num_hashes)

    signatures = []

    for vector in titles_1hot:
        signatures.append(get_signature(arr,vector))

    #merge signatures into single array
    signatures = np.stack(signatures)

    # Now for the actual LSH we will use a class

    from itertools import combinations

    class LSH:
        buckets = []
        counter = 0
        def __init__(self, b):
            self.b = b
            for i in range(b):
                self.buckets.append({})

        def make_subvecs(self, signature):
            l = len(signature)
            assert l % self.b == 0
            r = int(l / self.b)
            #break signature into subvectors
            subvecs = []
            for i in range(0, l, r):
                subvecs.append(signature[i:i+r])
            return np.stack(subvecs)

        def add_hash(self, signature):
            subvecs = self.make_subvecs(signature).astype(str)
            for i, subvec in enumerate(subvecs):
                subvec = ','.join(subvec)
                if subvec not in self.buckets[i].keys():
                    self.buckets[i][subvec] = []
                self.buckets[i][subvec].append(self.counter)
            self.counter += 1

        def check_candidates(self):
            candidates = []
            for bucket_band in self.buckets:
                keys = bucket_band.keys()
                for bucket in keys:
                    hits = bucket_band[bucket]
                    if len(hits) > 1:
                        candidates.extend(combinations(hits, 2))
            return set(candidates)

    # Generate candidate pairs
    b = buckets ### 64 buckets
    lsh = LSH(b)
    for signatures in signatures:
        lsh.add_hash(signatures)

    candidate_pairs = lsh.check_candidates()
    print(len(candidate_pairs))


    def diffBrand(candidate_pair: list):
        """"Check whether the items in a candidate pair belong to a different brand
        """
        idx1, idx2 = candidate_pair
        brand_1 = brands[idx1]
        brand_2 = brands[idx2]
        if brand_1 == 0 or brand_2 == 0:
            same_brand = 1
        elif brand_1 == brand_2:
            same_brand = 1
        else:
            same_brand = 0
        return same_brand

    def sameShop(candidate_pair: list):
        """Check whether the candidate pairs belong to the same shop
        """
        idx1, idx2 = candidate_pair
        shop_1 = shops[idx1]
        shop_2 = shops[idx2]
        if shop_1 == shop_2:
            same_shop = 1
        else:
            same_shop = 0
        return  same_shop

    def pair_to_key(candidate_pair: list):
        """This function takes as input a candidate pair and returns their KVP"""
        idx_1, idx_2 = candidate_pair
        kvp = features[idx_1], features[idx_2]
        return kvp

    def keyMatching(candidate_pair: list):
        """This function matches all the keys from one dictionary to another and
        gives the common keys as output. It also gives the non-matching keys of both
        products as output for later model words extraction
        """
        common_keys = []
        nmk_1 = []
        nmk_2 = []
        kvp_1, kvp_2 = pair_to_key(list(candidate_pair))
        for attribute in kvp_1.keys():
            if attribute in kvp_2.keys():
                common_keys.append(attribute)
            else:
                nmk_1.append({"key": attribute, "val": kvp_1[attribute]})

        for attribute in kvp_2.keys():
            if attribute not in kvp_1.keys():
                nmk_2.append({"key": attribute, "val": kvp_2[attribute]})
        return common_keys, nmk_1, nmk_2

    from Qgram import QGram

    def calcSim(key_1: str, key_2: str):
        """This function determines the similarity between shared KVP of a candidate pair
         using q-gram with q = 3, like in the paper of the prof stands"""
        if key_1 == key_2:
            return 1.0

        min_length = min(len(key_1), len(key_2))
        if min_length == 0:
            return 0
        if min_length > 3:
            min_length = 3

        qgram = QGram(min_length)
        dist, total = qgram.distance(key_1, key_2)
        sim = (total - dist) / total
        return sim

    def matching_model_words(set_a, set_b):
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

    def minFeatures(a, b):
        return min(len(a), len(b))

    gamma = 0.77
    epsilon = 0.67
    duplicate_count = 0
    real_duplicate_count = 0
    actual_real_duplicate_count = 0
    mu = 0.65 # TODO µ is the fixed weight of the TMWM similarity, if it returns a duplicate. We can leave it this way
    #

    count_alg_pairs = 0
    import numpy as np
    import time


    dist = np.zeros(shape=(len(titles),len(titles)))

    for i, candidate_pair in enumerate(candidate_pairs):
        if sameShop(candidate_pair) == 1 or diffBrand(candidate_pair) == 0:
            dist[candidate_pair[0],candidate_pair[1]] = 0
            # Not possible to be the same product
        else:
            count_alg_pairs += 1
            if keys[candidate_pair[0]] == keys[candidate_pair[1]]:
                actual_real_duplicate_count += 1


            #Run algorithm
            sim = 0
            avgSim = 0
            m = 0
            w = 0

            common_keys, nmk_1,nmk_2 = keyMatching(list(candidate_pair))

            for attribute_0 in nmk_1:
                for attribute_1 in nmk_2:
                    keySim = calcSim(attribute_0['key'], attribute_1['key'])
                    if keySim > gamma:
                        valueSim = calcSim(attribute_0['val'], attribute_1['val']) # requires values from the keys
                        weight = keySim
                        sim += weight*valueSim
                        m += 1
                        w += weight
                        # TODO check if removing the attribute is important.
                        # nmk_1.remove(attribute_0)
                        # nmk_2.remove(attribute_1)
            if w > 0:
                avgSim = sim / w

            kvp_1, kvp_2 = pair_to_key(candidate_pair)
            set_a = set([clean_model_value(model_word) for model_word in kvp_1.values()])
            set_b = set([clean_model_value(model_word) for model_word in kvp_2.values()])

            mwPerc = matching_model_words(set_a, set_b)

            from collections import Counter
            import math

            alpha = 0.40 # lower bound for titlesim
            beta = 0.60 # upper bound for titlesim

            counterA = Counter(process_title(titles[candidate_pair[0]]))
            counterB = Counter(process_title(titles[candidate_pair[1]]))
            # print(counterA,counterB)
            def counter_cosine_similarity(alpha, beta, c1, c2):
                """Calculates the cosine similarity between the model words in the title of product 1 and product 2
                of a candidate pair. Alpha and beta are threshold values that can be manually tuned."""

                terms = set(c1).union(c2)
                dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
                magA = math.sqrt(sum(c1.get(k, 0) ** 2 for k in terms))
                magB = math.sqrt(sum(c2.get(k, 0) ** 2 for k in terms))

                titleSim = dotprod / (magA * magB)
                if titleSim <= alpha:
                    titleSim = -1
                elif titleSim >= beta:
                    titleSim = 1

                return titleSim


            titleSim = counter_cosine_similarity(alpha, beta, counterA, counterB)


            if titleSim == -1:
                theta_1 = m / minFeatures(kvp_1, kvp_2)
                theta_2 = 1 - theta_1
                hSim = theta_1 * avgSim + theta_2 * mwPerc
                if hSim > epsilon:
                    duplicate_count +=1
                    if keys[candidate_pair[0]] == keys[candidate_pair[1]]:
                        real_duplicate_count += 1

            else:
                theta_1 = (1 - mu) * (m / minFeatures(kvp_1, kvp_2))
                theta_2 = 1 - mu - theta_1
                hSim = theta_1 * avgSim + theta_2 * mwPerc + mu * titleSim
                if hSim > epsilon:
                    duplicate_count += 1
                    if keys[candidate_pair[0]] == keys[candidate_pair[1]]:
                        real_duplicate_count += 1

    print(f"Number of candidate pairs = {len(candidate_pairs)}\n"
          f"Number of suspected duplicates = {duplicate_count}")
    print(f"Number of actual duplicates in considered = {real_duplicate_count}\n"
          f"Number of duplicates in canpairs = {actual_real_duplicate_count}")
    print(f"Number of pairs checked in algorithm = {count_alg_pairs}\n"
          f"Number of duplicates in dataset = {n_duplicates}")
    print("--------------------------------------------------------")

    # Results

    def results(dup_found, no_comp, dup_n):
        # LSH
        pair_quality = 2 * dup_found / no_comp
        pair_completeness = dup_found / dup_n
        f1_star = 2 * pair_quality * pair_completeness / (pair_quality + pair_completeness)
        precision = real_duplicate_count/duplicate_count
        recall = real_duplicate_count / n_duplicates
        f1 = 2 * precision * pair_completeness / (pair_quality + pair_completeness)
        possible_comp = len(titles) ** 2
        actual_comp = len(candidate_pairs)
        fraction_comp = actual_comp / possible_comp

        'The prof has for F1_star a value of around 0.02 and F1 of 0.4. Values around this are good, above is better, below is fine'


        return pair_quality, pair_completeness, f1_star, precision, recall , f1, fraction_comp
    #

    return results(dup_found=actual_real_duplicate_count, no_comp=len(candidate_pairs), dup_n=n_duplicates)

# experiment(buckets=1)





    # #NOT TODO TODO TODO. I will leave this out as it does not really impact the performance.
    # import matplotlib.pyplot as plt
    #
    # def hClustering(dist, epsilon):
    #     from sklearn.cluster import AgglomerativeClustering
    #     clustering = AgglomerativeClustering(affinity='precomputed', n_clusters = 2, linkage= 'complete').fit(dist)
    #     print(clustering.labels_)
    #     return clustering
    # label = hClustering(dist, epsilon)
    #
    # # Getting unique labels
    #
    # u_labels = np.unique(label)
    #
    # # plotting the results:
    #
    # for i in u_labels:
    #     plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    # plt.legend()
    # plt.show()
    #
