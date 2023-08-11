# batchris

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# Funções para a heurística de Christofides
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def christofides_tsp(cities):
    n = len(cities)

    # 1. Criar uma matriz de distâncias completa
    dist_matrix = np.array([[distance(cities[i], cities[j]) for j in range(n)] for i in range(n)])

    # 2. Encontrar uma MST usando a matriz de distâncias
    X = csr_matrix(dist_matrix)
    Tcsr = minimum_spanning_tree(X)
    mst = nx.from_scipy_sparse_array(Tcsr)

    # 3. Encontre os vértices de grau ímpar na MST
    odd_degree_nodes = [node for node, degree in dict(mst.degree()).items() if degree % 2 == 1]

    # 4. Formar pares mínimos de vértices de grau ímpar
    G = nx.Graph()
    G.add_weighted_edges_from((i, j, dist_matrix[i][j]) for i in odd_degree_nodes for j in odd_degree_nodes)
    matching = nx.max_weight_matching(G, maxcardinality=True)
    mst.add_edges_from(matching)

    # 5. Criar um circuito euleriano
    eulerian_circuit = list(nx.eulerian_circuit(mst))

    # 6. Converter o circuito euleriano em um caminho hamiltoniano
    hamiltonian_path = []
    visited = set()
    for u, v in eulerian_circuit:
        if u not in visited:
            visited.add(u)
            hamiltonian_path.append(u)
        if v not in visited:
            visited.add(v)
            hamiltonian_path.append(v)

    return hamiltonian_path
#

# Funções específicas do algoritmo do morcego

# Modificar a inicialização dos morcegos
def initialize_bats(n_bats, n_cities, initial_solution):
    bats = [np.random.permutation(n_cities) for _ in range(n_bats // 2)]

    # Adiciona morcegos com pequenas perturbações na solução inicial
    for _ in range(n_bats // 2):
        perturbed_bat = initial_solution[:-1].copy()
        perturbed_bat = swap_mutation(perturbed_bat)
        bats.append(perturbed_bat)

    freq = np.random.uniform(0, 2, n_bats)
    loudness = np.random.uniform(0.5, 1, n_bats)
    pulse_rate = np.random.uniform(0, 1, n_bats)
    return bats, freq, loudness, pulse_rate


def evaluate_solutions(bats, cities_dict):
    evaluations = [sum(distance(cities_dict[bats[i][j]], cities_dict[bats[i][j + 1]]) for j in range(len(bats[i]) - 1))
                   for i in range(len(bats))]
    return evaluations


def swap_mutation(bat):
    i, j = np.random.choice(len(bat) - 1, 2, replace=False) + 1
    bat[i], bat[j] = bat[j], bat[i]
    return bat

def reverse_sequence_mutation(bat):
    start, end = sorted(np.random.choice(len(bat) - 1, 2, replace=False) + 1)
    bat[start:end + 1] = bat[start:end + 1][::-1]
    return bat


def shuffle_mutation(bat):
    start, end = sorted(np.random.choice(len(bat) - 1, 2, replace=False) + 1)
    np.random.shuffle(bat[start:end + 1])
    return bat


# Modificar a função update_bats para usar ambas as operações de mutação
def update_bats(bats, best_bat, freq, loudness, pulse_rate, cities_dict):
    new_bats = []
    for idx, bat in enumerate(bats):
        if np.random.rand() > pulse_rate[idx]:
            if np.random.rand() < 0.5:
                bat = reverse_sequence_mutation(bat.copy())
            else:
                bat = swap_mutation(bat.copy())
        else:
            if np.random.rand() < loudness[idx]:
                bat = swap_mutation(bat.copy())

        # Ensure the bat is a valid route
        if bat[0] != 0:
            idx = np.where(bat == 0)[0][0]
            bat = np.concatenate((bat[idx:], bat[:idx]))

        new_bats.append(bat)
    return new_bats


def total_distance(path, cities_dict):
    return sum(distance(cities_dict[path[i]], cities_dict[path[i + 1]]) for i in range(len(path) - 1))


def bat_algorithm(cities_dict, n_bats=30, max_iter=100):
    n_cities = len(cities_dict)

    initial_solution = christofides_tsp(cities_dict)  # Calculando a solução inicial aqui.
    bats, freq, loudness, pulse_rate = initialize_bats(n_bats, n_cities,
                                                       initial_solution)  # Passando initial_solution como argumento.

    print("Solução Inicial (Christofides):", initial_solution)
    print("Custo Inicial:", total_distance(initial_solution, cities_dict))

    best_eval = float('inf')
    best_bat = None

    for iteration in range(max_iter):
        evaluations = evaluate_solutions(bats, cities_dict)
        current_best_eval = min(evaluations)
        if current_best_eval < best_eval:
            best_eval = current_best_eval
            best_bat = bats[np.argmin(evaluations)]

        bats = update_bats(bats, best_bat, freq, loudness, pulse_rate, cities_dict)


    final_solution = best_bat + [best_bat[0]]
    print("\nSolução Final:", final_solution)
    print("Custo Final:", total_distance(final_solution, cities_dict))

    return final_solution


# Definição das cidades (bays29) e execução
cities = list({
    (1150.0, 1760.0), (630.0, 1660.0), (40.0, 2090.0),
    (750.0, 1100.0), (750.0, 2030.0), (1030.0, 2070.0),
    (1650.0, 650.0), (1490.0, 1630.0), (790.0, 2260.0),
    (710.0, 1310.0), (840.0, 550.0), (1170.0, 2300.0),
    (970.0, 1340.0), (510.0, 700.0), (750.0, 900.0),
    (1280.0, 1200.0), (230.0, 590.0), (460.0, 860.0),
    (1040.0, 950.0), (590.0, 1390.0), (830.0, 1770.0),
    (490.0, 500.0), (1840.0, 1240.0), (1260.0, 1500.0),
    (1280.0, 790.0), (490.0, 2130.0), (1460.0, 1420.0),
    (1260.0, 1910.0), (360.0, 1980.0)
})

solution = bat_algorithm(cities)
print("Solução encontrada:", solution)

# Plotando as cidades e a solução
plt.scatter([city[0] for city in cities], [city[1] for city in cities], c='red')
for i in range(len(solution) - 1):
    plt.plot([cities[solution[i]][0], cities[solution[i+1]][0]],
             [cities[solution[i]][1], cities[solution[i+1]][1]], 'b-')
plt.show()
