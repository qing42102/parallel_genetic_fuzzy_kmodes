import numpy as np
import cupy as cp
from numba import njit, cuda
import time


def genetic_fuzzy_kmodes(
    data: np.ndarray,
    num_cluster: int,
    population_size: int = 10,
    alpha: float = 2,
    beta: float = 0.5,
    mutation_prob: float = 0.1,
    max_iter: int = 10,
):
    """
    Genetic fuzzy k-modes clustering algorithm
    """
    np.random.seed(0)

    chromosomes = initialize_population(population_size, num_cluster, data.shape[0])

    for i in range(max_iter):
        start = time.time()

        fitness = fitness_function(chromosomes, data, alpha, beta)

        chromosomes = selection(chromosomes, fitness)

        chromosomes = crossover(chromosomes, data, alpha)

        chromosomes = mutation(chromosomes, mutation_prob)

        print("Iteration", i, "time: ", time.time() - start)

    # Find the best chromosome in the last generation
    rank_index = rank_chromosomes(chromosomes, data, alpha)
    best_chromosome = chromosomes[rank_index][0]

    return best_chromosome


def rank_chromosomes(chromosomes: np.ndarray, data: np.ndarray, alpha: float):
    """
    Rank chromosomes based on their cost function
    """
    population_size = chromosomes.shape[0]

    cost = np.zeros(population_size)
    for i in range(population_size):
        centroids = calculate_centroids(chromosomes[i], data, alpha)
        cost[i] = cost_function(chromosomes[i], data, centroids, alpha)

    rank_index = np.argsort(cost)

    return rank_index


def fitness_function(
    chromosomes: np.ndarray, data: np.ndarray, alpha: float, beta: float
):
    """
    Fitness function for genetic algorithm based on the rank of the chromosome
    """
    rank_index = rank_chromosomes(chromosomes, data, alpha)

    rank = np.argsort(rank_index)

    fitness = beta * (np.power((1 - beta), rank))

    return fitness


def cost_function(
    cluster_membership: np.ndarray,
    data: np.ndarray,
    centroids: np.ndarray,
    alpha: float,
):
    """
    Cost function or objective function for fuzzy k-modes
    Sum of the number of the matches between data points and their respective centroids
    """
    num_clusters = cluster_membership.shape[1]

    cluster_membership = np.power(cluster_membership, alpha)

    cost = 0
    for i in range(num_clusters):
        # The cost is weighted by the cluster membership value
        cost += np.sum(
            cluster_membership[:, i] * dissimilarity_measure(data, centroids[i], axis=1)
        )

    return cost


def dissimilarity_measure(X: np.ndarray, Y: np.ndarray, axis: int = 0):
    """
    Distance function by calculating number of matches for integer arrays
    """

    return np.sum(X != Y, axis=axis)


def calculate_centroids(cluster_membership: np.ndarray, data: np.ndarray, alpha: float):
    """
    Calculate centroids for fuzzy k-modes
    """
    num_clusters = cluster_membership.shape[1]
    num_features = data.shape[1]

    centroids = np.zeros((num_clusters, num_features), dtype=int)

    cluster_membership = np.power(cluster_membership, alpha)

    for i in range(num_clusters):
        for j in range(num_features):
            # Count each category of the feature weighted by the cluster membership value
            weighted_count = np.bincount(data[:, j], weights=cluster_membership[:, i])

            # The index of the maximum value is the best category for that feature in the centroid
            centroids[i][j] = np.argmax(weighted_count)

    return centroids


def update_cluster_membership(centroids: np.ndarray, data: np.ndarray, alpha: float):
    """
    Update cluster membership based on the new centroids
    """
    num_data = data.shape[0]
    num_clusters = centroids.shape[0]

    cluster_membership = np.zeros((num_data, num_clusters))

    # The sum of the distance between the data point and all centroids
    distance_sum = np.zeros(num_data)
    for i in range(num_clusters):
        distance_sum += np.power(
            dissimilarity_measure(data, centroids[i], axis=1), 1 / (alpha - 1)
        )

    for i in range(num_clusters):
        # If the data matches all values of the centroid, the cluster membership is 1
        matches = np.all(data == centroids[i], axis=1)
        cluster_membership[:, i][matches] = 1

        # For the data that don't match, calculate the distance between the data point and the centroid
        # Divided by the sum of the distance
        distance_cluster = dissimilarity_measure(data, centroids[i], axis=1)
        cluster_membership[:, i] = (
            np.power(distance_cluster, 1 / (alpha - 1)) / distance_sum
        )

    return cluster_membership


def initialize_population(population_size: int, num_cluster: int, num_data: int):
    """
    Initialize a chromosme population with random values
    """
    random = np.random.rand(population_size, num_data, num_cluster)
    # Each row sums up to one
    chromosomes = random / np.sum(random, axis=2, keepdims=True)

    return chromosomes


def selection(chromosomes: np.ndarray, fitness: np.ndarray):
    """
    Selection of a new same-sized population of chromosomes
    Select from the current population based on the cumulative fitness
    """
    population_size = chromosomes.shape[0]

    cum_prob = np.cumsum(fitness)
    cum_prob = cum_prob / cum_prob[-1]

    new_chromosomes = np.zeros(chromosomes.shape)

    for i in range(population_size):
        random = np.random.rand(1)

        # Find the first chromosome with cum_prob > random
        index = np.where(cum_prob > random)[0][0]
        new_chromosomes[i] = chromosomes[index]

    return new_chromosomes


def crossover(chromosomes: np.ndarray, data: np.ndarray, alpha: float):
    """
    Crossover of chromosomes
    Uses one iteration of fuzzy k-modes
    """
    population_size = chromosomes.shape[0]

    new_chromosomes = np.zeros(chromosomes.shape)

    for i in range(population_size):
        centroids = calculate_centroids(chromosomes[i], data, alpha)
        cluster_membership = update_cluster_membership(centroids, data, alpha)
        new_chromosomes[i] = cluster_membership

    return new_chromosomes


def mutation(chromosomes: np.ndarray, mutate_prob: float):
    """
    Mutation of chromosomes by replacing some chromosomes randomly
    """
    population_size, num_data, num_clusters = chromosomes.shape

    for i in range(population_size):
        random = np.random.rand(num_data)

        # Use random number to determine which chromosome to mutate
        mutation_condition = random < mutate_prob
        num_mutations = np.count_nonzero(mutation_condition)

        if num_mutations > 0:
            # Generate new random values for the mutated chromosomes
            new_chromosomes = np.random.rand(num_mutations, num_clusters)
            new_chromosomes = new_chromosomes / np.sum(
                new_chromosomes, axis=1, keepdims=True
            )
            chromosomes[i][mutation_condition] = new_chromosomes

    return chromosomes
