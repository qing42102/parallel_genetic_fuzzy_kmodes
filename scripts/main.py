import numpy as np
from genetic_fuzzy_kmodes import genetic_fuzzy_kmodes
import os
from sklearn import metrics
import time


def load_data(file_path: str):
    data = np.load(file_path)
    print(data.shape)

    return data


def run_genetic_fuzzy_kmodes(
    data: np.ndarray,
    target: np.ndarray,
    num_cluster: int,
    population_size: int,
    alpha: float,
    beta: float,
    mutation_prob: float,
    max_iter: int,
):

    start_time = time.time()
    best_chromosome = genetic_fuzzy_kmodes(
        data,
        num_cluster=num_cluster,
        population_size=population_size,
        alpha=alpha,
        beta=beta,
        mutation_prob=mutation_prob,
        max_iter=max_iter,
    )
    print(best_chromosome)
    print("Total time:", time.time() - start_time)

    cluster = np.argmax(best_chromosome, axis=1)
    rand_score = metrics.adjusted_rand_score(target, cluster)
    print(rand_score)


if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    soy_data = load_data("../data/soy_data.npy")
    soy_target = load_data("../data/soy_target.npy")

    run_genetic_fuzzy_kmodes(
        soy_data,
        soy_target,
        num_cluster=4,
        population_size=20,
        alpha=1.2,
        beta=0.1,
        mutation_prob=0.01,
        max_iter=15,
    )

    metacritic_data = load_data("../data/metacritic_data.npy")
    metacritic_target = load_data("../data/metascore.npy")

    run_genetic_fuzzy_kmodes(
        metacritic_data,
        metacritic_target,
        num_cluster=7,
        population_size=50,
        alpha=1.2,
        beta=0.1,
        mutation_prob=0.01,
        max_iter=15,
    )
