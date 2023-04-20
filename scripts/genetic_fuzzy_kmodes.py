import numpy as np
import scipy as sp
import cupy as cp
from numba import jit, cuda
from collections import defaultdict

def genetic_fuzzy_kmodes(data: np.ndarray, num_cluster: int, population_size: int, alpha: float = 2, beta: float = 0.5):
    np.random.seed(0)

    chromosomes = initialize_population(population_size, num_cluster, data)
    
    chromosomes = selection(chromosomes, data, alpha, beta)
    
    # crossover()
    
    # mutation()


    return

def initialize_population(population_size: int, num_cluster: int, data: np.ndarray):
    num_data = data.shape[0]

    random = np.random.rand(population_size, num_data, num_cluster)
    chromosomes = random / np.sum(random, axis=2, keepdims=True)

    return chromosomes

def fitness_function(chromosomes: np.ndarray, data: np.ndarray, alpha: float, beta: float):
    population_size = chromosomes.shape[0]

    cost = np.zeros(population_size)
    for i in range(population_size):
        centroids = calculate_centroids(chromosomes[i], data, alpha)
        cost[i] = cost_function(chromosomes[i], data, centroids, alpha)

    rank = cost.argsort().argsort() + 1

    fitness = beta*(np.power((1 - beta), rank))

    return fitness

def cost_function(cluster_membership: np.ndarray, data: np.ndarray, centroids: np.ndarray, alpha: float):
    '''
    Cost function or objective function for fuzzy k-modes
    '''
    num_data, num_clusters = cluster_membership.shape

    cost = 0
    for i in range(num_clusters):
        for j in range(num_data):
            cost += np.power(cluster_membership[j][i], alpha) * dissimilarity_measure(data[j], centroids[i])

    return cost

def dissimilarity_measure(X, Y): 
    
    return np.sum(X!=Y, axis = 0)

def calculate_centroids(cluster_membership: np.ndarray, data: np.ndarray, alpha: float):
    '''
    Calculate centroids for fuzzy k-modes
    '''
    num_data, num_clusters = cluster_membership.shape
    num_features = data.shape[1]
    
    centroids = np.zeros((num_clusters, num_features))

    cluster_membership = np.power(cluster_membership, alpha)

    for z in range(num_clusters):
        for x in range(num_features):
            freq = defaultdict(int)
            for y in range(num_data):
                freq[data[y][x]] += cluster_membership[y][z]

            centroids[z][x] = max(freq, key = freq.get)
    
    centroids = centroids.astype(int)

    return centroids

def selection(chromosomes: np.ndarray, data: np.ndarray, alpha: float, beta: float):
    population_size = chromosomes.shape[0]
    fitness = fitness_function(chromosomes, data, alpha, beta)

    cum_prob = np.cumsum(fitness)
    cum_prob = cum_prob / cum_prob[-1]

    new_chromosomes = np.zeros(chromosomes.shape)

    for i in range(population_size):
        # Random number to pick chromosome
        random = np.random.rand(1)

        for j in range(population_size):
            # Pick chromosome
            if random < cum_prob[j]:
                new_chromosomes[i] = chromosomes[j]
                break
            
    return new_chromosomes

def crossover():

    return

def mutation():

    return