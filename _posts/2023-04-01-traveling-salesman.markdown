---
title:  "Traveling salesman problem"
date:   2023-04-01 16:00:00
categories: [Tech, Algorithms] 
tags: [tsp, algorithms, python]
---

The traveling salesman problem is one of those annoying interview questions you sometimes get hurled towards you, accompanied with "how would you solve this"? I remember a time where I was 
a fresh hatchling straight out of college, finding myself in that exact situation. I'll spare you the details, but I can tell you I didn't get particularly far ;-)

Times change though, and over the years I definitely grew into the type of software engineer that definitely prefers to work with data structures and algorithms over the next "best" framework to solve trivial problems like yeeting text and images on a screen. 

![salesmen]({{ site.url }}/assets/tsp/salesmen.jpg)

### Some solutions...

So, quite a few years later, I thought it'd be a nice moment to provide a few potential solutions, and give you a little bit of a summary regarding how to go about it. I assume you know what the [Traveling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem){:target="_blank"} entails, at least from a high level. It's one of those nice annoying NP-hard ones, so more than enough ways to have fun with it. In this article, we'll go over 4 example implementations of possible solutions to the problem:
* a simple brute force algorithm, guaranteed to give you the optimal shortest distance, but obviously horribly slow
* another guaranteed optimal solution, but quite a bit more efficient, using dynamic programming
* an approximation of a solution using genetic algorithms
* another approximation, using ant colony optimization

I've been interested in genetic algorithms and ant colony optimization for a while now, and the traveling salesman problem is a nice use-case to apply these algorithms to. In case you are not familiar with one or either of them, you can find more information about [Genetic algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm){:target="_blank} and [Ant Colony Optimization](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms){:target="_blank} all over the internet. Or you can ask ChatGPT, apparently that's quite the trend lately.


### The project
The full code of the project is available on my [GitHub](https://github.com/tomvanschaijk/travelingsalesman){:target="_blank"}. The requirements to run it are pretty basic. The code is all in Python, and I mainly use PyGame, Numpy, Asyncio and Aiostream. Just install the requirements in requirements.txt and you'll be set. You can preview the end result right [here](https://youtu.be/XCZSwM--vCA){:target="_blank"}. In fact, give that a quick gander and the following summary will quickly become clear.

The PyGame window that pops up when running simply consists of 5 panes. The top center one is the one where you can simple click to add points. These points will be the destination our salesman will have to visit. The first point you create will be green, and will be the starting point and ending destination. All other blue ones are the cities to be visited before getting back to the starting point. Hitting spacebar will reset the screen, enter starts the algorithm.

Since the brute force and dynamic programming solutions are inherently slow for anything more than a few points, you'll notice there's a cut-off point where those algorithms are not being taken along for the ride anymore. It would simply take more time, as the time complexity for both just explodes. Dynamic programming will be used for a bit longer than brute force, but when you start getting in the double digits in terms of destination count, both will not be executed anymore.

In case you wish to follow along in the code and you check out the GitHub project, there's not a lot of files of interest:
* main.py: nothing you wouldn't expect. The only code of slight interest in there is how to run the 4 algorithms concurrently. All the rest is setup for PyGame, running the main "game loop".
* graph.py: contains an implementation of an undirected weighted graph, slightly tailored to the current problem at hand
* the folder /solvers contains the implementations of the 4 algorithms. The rest of the article will mostly focus on those.

### Processing the results
As stated earlier, the main.py file doesn't contain much of interest regarding the actual problem. However, maybe one slight little thing to touch on is how the 4 algorithms are executed concurrently, and results are processed as they come in.

``` python
algorithms = [algorithm for algorithm in self.__algorithms
              if algorithm is not None]
zipped = stream.ziplatest(*algorithms)
merged = stream.map(zipped, lambda x: dict(enumerate(x)))
async with merged.stream() as streamer:
    async for resultset in streamer:

```

To explain the code, it might be nice to focus on what goes on in there, using another example:

``` python
import asyncio
from aiostream import stream


async def process_results_interleaved(races):
    combine = stream.merge(*races)
    async with combine.stream() as streamer:
        async for item in streamer:
            print(item)


async def process_results_packet(races):
    zipped = stream.ziplatest(*races)
    merged = stream.map(zipped, lambda x: dict(enumerate(x)))
    async with merged.stream() as streamer:
        async for resultset in streamer:
            print(resultset)


async def race(racer, sleep_time, checkpoints):
    for i in range(1, checkpoints + 1):
        await asyncio.sleep(sleep_time)
        if i == checkpoints:
            yield f"{racer} finished!"
        else:
            yield f"{racer} hits checkpoint: {i}"


def create_races(checkpoints):
    return [race("Turtle", 2, checkpoints),
            race("Hare", 1, checkpoints),
            race("Dragster", 0.3, checkpoints)]


def main():
    checkpoints = 10

    print("Starting race, processing results as they come in from each contestant:")
    races = create_races(checkpoints)
    asyncio.run(process_results_interleaved(races))

    print("Starting race, processing results in packets:")
    races = create_races(checkpoints)
    asyncio.run(process_results_packet(races))


if __name__ == "__main__":
    main()
```

This little script gives you the following output:

![race_results]({{ site.url }}/assets/tsp/race_results.png)


### A brute force approach


``` python
async def brute_force(graph: Graph, distances: dict[tuple[int, int], int]) -> AsyncIterator[AlgorithmResult]:
    """Solve the TSP problem with a brute force implementation, running through all permutations"""
    unique_permutations = set()
    paths_evaluated = 0
    evaluations_until_solved = 0
    start = graph[0].key
    keys = list(graph.nodes.keys())
    sub_keys = keys[1:]
    for permutation in permutations(sub_keys):
        permutation = (start,) + permutation + (start,)
        if permutation not in unique_permutations and permutation[::-1] not in unique_permutations:
            paths_evaluated += 1
            unique_permutations.add(permutation)
            current_path_length = 0
            node = start
            vertices: list[tuple[int, int, int]] = []
            for key in permutation[1:]:
                distance = distances[(node, key)]
                current_path_length += distance
                vertices.append((node, key, distance))
                node = key

            graph.remove_vertices()
            graph.add_vertices(vertices)
            if current_path_length < graph.optimal_cycle_length:
                graph.optimal_cycle = ShortestPath(current_path_length, vertices)
                evaluations_until_solved = paths_evaluated
            await asyncio.sleep(0.0001)
            yield AlgorithmResult(paths_evaluated, evaluations_until_solved)

    graph.remove_vertices()
    graph.add_vertices(graph.optimal_cycle.vertices)
    yield AlgorithmResult(paths_evaluated, evaluations_until_solved)
```


### Dynamic programming to the rescue


``` python
async def dynamic_programming(graph: Graph, distances: dict[tuple[int, int], int]
                              ) -> AsyncIterator[AlgorithmResult]:
    """Solve the TSP problem with dynamic programming"""
    node_count = len(graph)
    start = graph[0].key
    memo = [[0 for _ in range(1 << node_count)] for __ in range(node_count)]
    optimal_cycle = []
    optimal_cycle_length = maxsize
    cycles_evaluated = 0
    evaluations_until_solved = 0
    memo = setup(memo, graph, distances, start)
    for nodes_in_subcycle in range(3, node_count + 1):
        for subcycle in initialize_combinations(nodes_in_subcycle, node_count):
            if is_not_in(start, subcycle):
                continue

            cycles_evaluated += 1
            # Look for the best next node to attach to the cycle
            for next_node in range(node_count):
                if next_node == start or is_not_in(next_node, subcycle):
                    continue

                subcycle_without_next_node = subcycle ^ (1 << next_node)
                min_cycle_length = maxsize
                for last_node in range(node_count):
                    if (last_node == start or last_node == next_node or is_not_in(last_node, subcycle)):
                        continue

                    new_cycle_length = (memo[last_node][subcycle_without_next_node] + distances[(last_node, next_node)])
                    if new_cycle_length < min_cycle_length:
                        min_cycle_length = new_cycle_length
                    memo[next_node][subcycle] = min_cycle_length

        evaluations_until_solved = cycles_evaluated
        optimal_cycle_length = calculate_optimal_cycle_length(start, nodes_in_subcycle, memo, distances)
        optimal_cycle = find_optimal_cycle(start, nodes_in_subcycle, memo, distances)
        vertices = create_vertices(optimal_cycle, distances)
        graph.optimal_cycle = ShortestPath(optimal_cycle_length, vertices)
        await asyncio.sleep(0.0001)
        yield AlgorithmResult(cycles_evaluated, evaluations_until_solved)

    optimal_cycle_length = calculate_optimal_cycle_length(start, node_count, memo, distances)
    optimal_cycle = find_optimal_cycle(start, node_count, memo, distances)
    vertices = create_vertices(optimal_cycle, distances)
    graph.remove_vertices()
    graph.add_vertices(vertices)
    graph.optimal_cycle = ShortestPath(optimal_cycle_length, vertices)
    yield AlgorithmResult(cycles_evaluated, evaluations_until_solved)


def setup(memo: list, graph: Graph, distances: dict[tuple[int, int], int], start: int) -> list:
    """Prepare the array used for memoization during the dynamic programming algorithm"""
    for i, node in enumerate(graph):
        if start == node.key:
            continue

        memo[i][1 << start | 1 << i] = distances[(start, i)]

    return memo


def initialize_combinations(nodes_in_subcycle: int, node_count: int) -> list[int]:
    """Initialize the combinations to consider in the next step of the algorithm"""
    subcycle_list = []
    initialize_combination(0, 0, nodes_in_subcycle, node_count, subcycle_list)

    return subcycle_list


def initialize_combination(subcycle, at, nodes_in_subcycle, node_count, subcycle_list) -> None:
    """Initialize the combination to consider in the next step of the algorithm"""
    elements_left_to_pick = node_count - at
    if elements_left_to_pick < nodes_in_subcycle:
        return

    if nodes_in_subcycle == 0:
        subcycle_list.append(subcycle)
    else:
        for i in range(at, node_count):
            subcycle |= 1 << i
            initialize_combination(subcycle, i + 1, nodes_in_subcycle - 1,
                                   node_count, subcycle_list)
            subcycle &= ~(1 << i)


def is_not_in(index, subcycle) -> bool:
    """Checks if the bit at the given index is a 0"""
    return ((1 << index) & subcycle) == 0


def calculate_optimal_cycle_length(start: int, node_count: int, memo: list,
                                   distances: dict[tuple[int, int], int]) -> int:
    """Calculate the optimal cycle length"""
    end = (1 << node_count) - 1
    optimal_cycle_length = maxsize
    for i in range(node_count):
        if i == start:
            continue

        cycle_cost = memo[i][end] + distances[(i, start)]
        if cycle_cost < optimal_cycle_length:
            optimal_cycle_length = cycle_cost

    return optimal_cycle_length


def find_optimal_cycle(start: int, node_count: int, memo: list,
                       distances: dict[tuple[int, int], int]):
    """Recreate the optimal cycle"""
    last_index = start
    state = (1 << node_count) - 1
    optimal_cycle: list[int] = []
    for _ in range(node_count - 1, 0, -1):
        index = -1
        for j in range(node_count):
            if j == start or is_not_in(j, state):

                continue
            if index == -1:
                index = j
            prev_cycle_length = memo[index][state] + distances[(index, last_index)]
            new_cycle_length = memo[j][state] + distances[(j, last_index)]
            if new_cycle_length < prev_cycle_length:
                index = j

        optimal_cycle.append(index)
        state = state ^ (1 << index)
        last_index = index
    optimal_cycle.append(start)
    optimal_cycle.reverse()
    optimal_cycle.append(start)

    return optimal_cycle


def create_vertices(optimal_cycle: list[int], distances: dict[tuple[int, int], int]
                    ) -> list[tuple[int, int, int]]:
    """Transform the list of visited node keys to something our graph can work with"""
    vertices: list[tuple[int, int, int]] = []
    for i in range(1, len(optimal_cycle)):
        weight = distances[(optimal_cycle[i - 1], optimal_cycle[i])]
        vertices.append((optimal_cycle[i - 1], optimal_cycle[i], weight))

    return vertices
```


### Genetic algorithm

This is where the fun starts...

``` python
async def genetic_algorithm(graph: Graph, distances: dict[tuple[int, int], int], population_size: int,
                            max_generations: int, max_no_improvement: int) -> AsyncIterator[AlgorithmResult]:
    """Solve the TSP problem with a genetic algorithm"""
    generations_evaluated = 0
    generations_until_solved = 0
    generations_without_improvement = 0
    optimal_cycle: list[int] = []
    optimal_cycle_length = maxsize
    population = spawn(graph, population_size)
    cycle_lengths = get_cycle_lengths(population, distances)
    for _ in range(max_generations):
        if generations_without_improvement >= max_no_improvement:
            break

        fitness = determine_fitness(cycle_lengths)
        improved = False
        for i, cycle_length in enumerate(cycle_lengths):
            if cycle_length < optimal_cycle_length:
                improved = True
                optimal_cycle_length = cycle_length
                optimal_cycle = population[i]

        generations_evaluated += 1
        if improved:
            generations_until_solved = generations_evaluated
            generations_without_improvement = 0
            vertices: list[tuple[int, int, int]] = []
            node = optimal_cycle[0]
            for key in optimal_cycle[1:]:
                distance = distances[(node, key)]
                vertices.append((node, key, distance))
                node = key
            graph.remove_vertices()
            graph.add_vertices(vertices)
            graph.optimal_cycle = ShortestPath(optimal_cycle_length, vertices)
        else:
            generations_without_improvement += 1

        if len(population) == 1:
            break
        population, cycle_lengths = create_next_population(population, cycle_lengths,
                                                           fitness, distances)
        await asyncio.sleep(0.0001)
        yield AlgorithmResult(generations_evaluated, generations_until_solved)
    yield AlgorithmResult(generations_evaluated, generations_until_solved)


def spawn(graph: Graph, population_size: int) -> list[list[int]]:
    """Create the initial generation"""
    start = graph[0].key
    keys = list(graph.nodes.keys())[1:]
    max_size = int(factorial(len(keys)) / 2)
    unique_permutations = set()
    while len(unique_permutations) < population_size and len(unique_permutations) < max_size:
        permutation = list(keys)
        shuffle(permutation)
        permutation = (start,) + tuple(permutation) + (start,)
        if permutation[::-1] not in unique_permutations:
            unique_permutations.add(permutation)

    return [list(permutation) for permutation in unique_permutations]


def create_next_population(current_population: list[list[int]], cycle_lengths: list[int],
                           fitness: list[float], distances: dict[tuple[int, int], int]
                           ) -> tuple[list[list[int]], list[int]]:
    """Create the next generation"""
    new_population: list[list[int]] = []
    population_size = len(current_population)

    # Create the offspring of the current generation
    offspring = create_offspring(current_population, fitness, population_size)

    # Perform a variation of elitism where we add the offspring to the current generation
    # and only continue with the fittest list of size population_size
    new_population = current_population + offspring
    offspring_cycle_lengths = get_cycle_lengths(offspring, distances)
    new_population_cycle_lengths = cycle_lengths + offspring_cycle_lengths
    new_population_fitness = fitness + determine_fitness(offspring_cycle_lengths)
    survivor_candidates = zip(new_population_fitness, new_population)
    fittest_indices = [i for _, i in heapq.nlargest(population_size, ((x, i) for i, x in enumerate(survivor_candidates)))]
    new_population = [new_population[i] for i in fittest_indices]
    new_population_cycle_lengths = [new_population_cycle_lengths[i] for i in fittest_indices]

    return new_population, new_population_cycle_lengths


def create_offspring(current_population: list[list[int]], fitness: list[float],
                     population_size: int) -> list[list[int]]:
    """Create a new generation"""
    offspring: list[list[int]] = []
    while len(offspring) < population_size:
        parent1 = parent2 = 0
        while parent1 == parent2:
            parent1 = get_parent(fitness)
            parent2 = get_parent(fitness)

        child1 = crossover(current_population[parent1], current_population[parent2])
        child2 = crossover(current_population[parent2], current_population[parent1])

        child1 = mutate(child1)
        child2 = mutate(child2)

        offspring.append(child1)
        offspring.append(child2)

    return offspring


def get_parent(fitness: list[float]):
    """Get a parent using either tournament selection or biased random selection"""
    if randint(0, 1):
        return tournament_selection(fitness)

    return biased_random_selection(fitness)


def tournament_selection(fitness: list[float]) -> int:
    """Perform basic tournament selection to get a parent"""
    start, end = 0, len(fitness) - 1
    candidate1 = randint(start, end)
    candidate2 = randint(start, end)
    while candidate1 == candidate2:
        candidate2 = randint(start, end)

    return candidate1 if fitness[candidate1] > fitness[candidate2] else candidate2


def biased_random_selection(fitness: list[float]) -> int:
    """Perform biased random selection to get a parent"""
    random_specimen = randint(0, len(fitness) - 1)
    for i, _ in enumerate(fitness):
        if fitness[i] >= fitness[random_specimen]:
            return i

    return random_specimen


def crossover(parent1: list[int], parent2: list[int]) -> list[int]:
    """Cross-breed a new set of children from the given parents"""
    start = parent1[0]
    end = parent1[len(parent1) - 1]
    parent1 = parent1[1:len(parent1) - 1]
    parent2 = parent2[1:len(parent2) - 1]
    split = randint(1, len(parent1) - 1)
    child: list[int] = [0] * len(parent1)
    for i in range(split):
        child[i] = parent1[i]

    remainder = [i for i in parent2 if i not in child]
    for i, data in enumerate(remainder):
        child[split + i] = data

    return [start, *child, end]


def mutate(child: list[int]) -> list[int]:
    """Mutate the child sequence"""
    if randint(0, 1):
        child = swap_mutate(child)
    child = rotate_mutate(child)

    return child


def swap_mutate(child: list[int]) -> list[int]:
    """Mutate the cycle by swapping 2 nodes"""
    index1 = randint(1, len(child) - 2)
    index2 = randint(1, len(child) - 2)
    child[index1], child[index2] = child[index2], child[index1]

    return child


def rotate_mutate(child: list[int]) -> list[int]:
    """Mutate the cycle by rotating a part nodes"""
    split = randint(1, len(child) - 2)
    head = child[0:split]
    mid = child[split:len(child) - 1][::-1]
    tail = child[len(child) - 1:]
    child = head + mid + tail

    return child


def get_cycle_lengths(population: list[list[int]],
                      distances: dict[tuple[int, int], int]) -> list[int]:
    """Get the lengths of all cycles in the graph"""
    cycle_lengths: list[int] = []
    for specimen in population:
        node = specimen[0]
        cycle_length = 0
        for key in specimen[1:]:
            key = int(key)
            cycle_length += distances[(node, key)]
            node = key
        cycle_lengths.append(cycle_length)

    return cycle_lengths


def determine_fitness(cycle_lengths: list[int]) -> list[float]:
    """Determine the fitness of the specimens in the population"""
    # Invert so that shorter paths get higher values
    fitness_sum = sum(cycle_lengths)
    fitness = [fitness_sum / cycle_length for cycle_length in cycle_lengths]

    # Normalize the fitness
    fitness_sum = sum(fitness)
    fitness = [f / fitness_sum for f in fitness]

    return fitness
```

### Thousands of little creepers...

``` python
async def ant_colony(graph: Graph, distances: dict[tuple[int, int], int],
                     max_swarms: int, max_no_improvement: int) -> AsyncIterator[AlgorithmResult]:
    """Solve the TSP problem using ant colony optimization"""
    swarms_evaluated = 0
    evaluations_until_solved = 0
    swarms_without_improvement = 0
    node_count = len(graph.nodes)
    pheromones = initialize_pheromones(node_count, distances)
    for _ in range(max_swarms):
        if swarms_without_improvement >= max_no_improvement:
            break

        vertices, cycle_lengths = swarm_traversal(pheromones, distances)
        pheromones = pheromone_evaporation(pheromones)
        best_cycle_index = np.argmin(cycle_lengths)
        best_cycle = vertices[best_cycle_index]
        best_cycle_length = cycle_lengths[best_cycle_index]
        pheromones = pheromone_release(vertices, best_cycle, best_cycle_length, pheromones)
        pheromones = normalize(pheromones)
        swarms_evaluated += 1

        if cycle_lengths[best_cycle_index] < graph.optimal_cycle_length:
            graph.remove_vertices()
            graph.add_vertices(best_cycle)
            graph.optimal_cycle = ShortestPath(best_cycle_length, best_cycle)
            evaluations_until_solved = swarms_evaluated
            swarms_without_improvement = 0
            await asyncio.sleep(0.0001)
            yield AlgorithmResult(swarms_evaluated, evaluations_until_solved)
        else:
            swarms_without_improvement += 1
    yield AlgorithmResult(swarms_evaluated, evaluations_until_solved)


def initialize_pheromones(node_count: int, distances: dict[tuple[int, int], int]) -> np.ndarray:
    """Initialize the pheromone array"""
    pheromones = np.zeros((node_count, node_count), dtype=float)
    for i, j in np.ndindex(pheromones.shape):
        if i != j:
            pheromones[i, j] = distances[(i, j)]
    for i in range(node_count):
        row_sum = sum(pheromones[i, :])
        for j in range(node_count):
            if i != j:
                pheromones[i, j] = row_sum / pheromones[i, j]
        row_sum = sum(pheromones[i, :])
        for j in range(node_count):
            if i != j:
                pheromones[i, j] /= row_sum
    return pheromones


def normalize(pheromones: np.ndarray) -> np.ndarray:
    """Normalize the pheromone matrix into probabilities"""
    node_count = pheromones.shape[0]
    for i in range(node_count):
        row_sum = sum(pheromones[i, :])
        for j in range(node_count):
            if i != j:
                pheromones[i, j] /= row_sum
    return pheromones


def swarm_traversal(pheromones: np.ndarray, distances: dict[tuple[int, int], int]) -> tuple[np.ndarray, list[int]]:
    """Traverse the graph with a number of ants equal to the number of nodes"""
    node_count = pheromones.shape[0]
    swarm_size = node_count * node_count
    vertices = np.array([[(0, 0, 0)] * node_count] * swarm_size, dtype="i,i,i")
    cycle_lengths: list[int] = [0] * swarm_size
    # Traverse the graph swarm_size times
    with ThreadPoolExecutor(max_workers=min(swarm_size, 50)) as executor:
        futures = [executor.submit(traverse, node_count, pheromones, distances)
                   for i in range(swarm_size)]
        for i, completed in enumerate(as_completed(futures)):
            cycle_length, cycle = completed.result()
            vertices[i] = cycle
            cycle_lengths[i] = cycle_length
    return vertices, cycle_lengths


def traverse(node_count: int, pheromones: np.ndarray,
             distances: dict[tuple[int, int], int]) -> tuple[int, np.ndarray]:
    """Perform a traversal through the graph"""
    # Each traversal consists of node_count vertices
    current_node = 0
    visited = set([current_node])
    cycle_length = 0
    vertices = np.array([(0, 0, 0)] * node_count, dtype="i,i,i")
    for j in range(node_count - 1):
        row_sorted_indices = np.argsort(pheromones[current_node, :])
        row = np.take(pheromones[current_node, :], row_sorted_indices)
        cumul = 0
        for k, _ in enumerate(row):
            if row[k] == 0.0:
                continue
            row[k], cumul = row[k] + cumul, row[k] + cumul

        index = -1
        chance = random()
        for k in range(1, len(row)):
            candidate = row_sorted_indices[k]
            if (row[k - 1] < chance <= row[k] and candidate != current_node and candidate not in visited):
                index = candidate
                break
        # If no suitable index was found, the generated chance was probably too low
        # Pick the first index that's not itself and not visited yet
        if index == -1:
            for k in range(len(row) - 1, 0, -1):
                candidate = row_sorted_indices[k]
                if candidate != current_node and candidate not in visited:
                    index = candidate
                    break

        distance = distances[current_node, index]
        vertices[j] = (current_node, index, distance)
        cycle_length += distance
        visited.add(index)
        current_node = index
    # Add the last vertex back to the starting node
    distance = distances[current_node, 0]
    vertices[node_count - 1] = (current_node, 0, distance)
    cycle_length += distance
    return cycle_length, vertices


def pheromone_evaporation(pheromones: np.ndarray) -> np.ndarray:
    """Evaporation of pheromones after each traversal"""
    for index in np.ndindex(pheromones.shape):
        pheromones[index] *= 1 - pheromones[index]
    return pheromones


def pheromone_release(vertices: np.ndarray, best_cycle: np.ndarray,
                      best_cycle_length: int, pheromones: np.ndarray) -> np.ndarray:
    """Perform pheromone release, with elitism towards shorter cycles"""
    for cycle in vertices:
        for i, j, weight in cycle:
            pheromones[i, j] += 1 / weight
    for i, j, _ in best_cycle:
        pheromones[i, j] += 1 / best_cycle_length

    return pheromones
```