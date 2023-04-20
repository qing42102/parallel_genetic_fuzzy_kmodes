import numpy as np
from genetic_fuzzy_kmodes import genetic_fuzzy_kmodes
import os
from sklearn.datasets import load_iris

def load_data(file_path: str):
    data = np.load(file_path)
    print(data.shape)

    return data

if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # data = load_data("../data/data.npy")
    # target = load_data("../data/metascore.npy")

    (data, target) = load_iris(return_X_y=True)
    
    genetic_fuzzy_kmodes(data, 3, 5)
