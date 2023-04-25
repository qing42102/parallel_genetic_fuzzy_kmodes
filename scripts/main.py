import numpy as np
from genetic_fuzzy_kmodes import genetic_fuzzy_kmodes
import os
from sklearn import metrics
import time

def load_data(file_path: str):
    data = np.load(file_path)
    print(data.shape)

    return data

if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # data = load_data("../data/metacritic_data.npy")
    # target = load_data("../data/metascore.npy")

    soy_data = load_data("../data/soy_data.npy")
    soy_target = load_data("../data/soy_target.npy")
    
    start_time = time.time()
    best_chromosome = genetic_fuzzy_kmodes(soy_data, num_cluster=4, population_size=20, alpha=1.2, beta=0.1, mutation_prob=0.01, max_iter=15)
    print(best_chromosome)
    print("Time", time.time() - start_time)

    cluster = np.argmax(best_chromosome, axis=1)
    rand_score = metrics.adjusted_rand_score(soy_target, cluster)
    print(rand_score)