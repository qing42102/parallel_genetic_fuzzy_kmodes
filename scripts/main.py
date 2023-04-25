import numpy as np
from genetic_fuzzy_kmodes import genetic_fuzzy_kmodes
import os

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
    
    best_chromosome = genetic_fuzzy_kmodes(soy_data, 4, 5)
    print(best_chromosome)
