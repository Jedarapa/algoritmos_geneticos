import random
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

# ========================================================
# 1. Plantilla de datos de prueba (más compleja)
# ========================================================
template = {
    "int_param": {"type": "int", "min": -20, "max": 20},
    "float_param": {"type": "float", "min": -10.0, "max": 10.0},
    "choice_param": {"type": "choice", "options": ["red", "green", "blue", "yellow"]},
    "string_param": {"type": "string", "length": 5, "chars": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    "nested_param": {
        "type": "nested",
        "schema": {
            "subint": {"type": "int", "min": -20, "max": 20},
            "subfloat": {"type": "float", "min": -5.0, "max": 5.0},
            "sublist": {"type": "list", "length": 3, "element": {"type": "float", "min": 0.0, "max": 1.0}}
        }
    }
}

# ========================================================
# 2. Funciones objetivo para evaluar el caso de prueba
# ========================================================
def target_function_real(chromosome):
    a = chromosome.genes["int_param"]
    b = chromosome.genes["nested_param"]["subint"]
    branches = set()
    branch_distances = {}
    if a > 0:
        branches.add("a_positive")
        da = 0
    else:
        branches.add("a_nonpositive")
        da = abs(a) / 20.0
    branch_distances["a"] = da
    if b > 0:
        branches.add("b_positive")
        db = 0
    else:
        branches.add("b_nonpositive")
        db = abs(b) / 20.0
    branch_distances["b"] = db
    if a > 0 and b > 0:
        if a > b:
            branches.add("a_gt_b")
            dab = 0
        else:
            branches.add("a_not_gt_b")
            dab = (b - a) / (abs(b) + 1)
    else:
        branches.add("not_applicable")
        dab = 1.0
    branch_distances["ab"] = dab
    total_distance = da + db + dab
    fitness = 1.0 / (1.0 + total_distance)
    cov = ((1 if da==0 else 0) + (1 if db==0 else 0) + (1 if dab==0 else 0)) / 3.0
    return branches, total_distance, branch_distances, fitness, cov

def target_function_complex(chromosome):
    a = chromosome.genes["int_param"]
    b = chromosome.genes["float_param"]
    choice = chromosome.genes["choice_param"]
    s = chromosome.genes["string_param"]
    c = chromosome.genes["nested_param"]["subint"]
    d = chromosome.genes["nested_param"]["subfloat"]
    L = chromosome.genes["nested_param"]["sublist"]
    avg_L = sum(L) / len(L)
    branches = set()
    branch_distances = {}
    total_possible = 6
    # B1: a > 0 y c > 0
    if a > 0 and c > 0:
        branches.add("B1_true")
        d1 = 0
    else:
        branches.add("B1_false")
        d1 = (abs(min(a, 0))/20.0) + (abs(min(c, 0))/20.0)
    branch_distances["B1"] = d1
    # B2: a + b > 0
    if a + b > 0:
        branches.add("B2_true")
        d2 = 0
    else:
        branches.add("B2_false")
        d2 = (0 - (a + b)) / 30.0
    branch_distances["B2"] = d2
    # B3: choice in {"red", "green"}
    if choice in ["red", "green"]:
        branches.add("B3_true")
        d3 = 0
    else:
        branches.add("B3_false")
        d3 = 1.0
    branch_distances["B3"] = d3
    # B4: s tiene longitud 5 y s[0] en "ABC"
    if len(s) == 5 and s[0] in "ABC":
        branches.add("B4_true")
        d4 = 0
    else:
        branches.add("B4_false")
        d4 = 1.0
    branch_distances["B4"] = d4
    # B5: promedio de L > 0.5
    if avg_L > 0.5:
        branches.add("B5_true")
        d5 = 0
    else:
        branches.add("B5_false")
        d5 = (0.5 - avg_L) / 0.5 if avg_L < 0.5 else 0
    branch_distances["B5"] = d5
    # B6: d < 0
    if d < 0:
        branches.add("B6_true")
        d6 = 0
    else:
        branches.add("B6_false")
        d6 = d / 5.0
    branch_distances["B6"] = d6
    total_distance = d1 + d2 + d3 + d4 + d5 + d6
    coverage = sum(1 for key in branch_distances if branch_distances[key] == 0) / total_possible
    fitness = 0.5 * coverage + 0.5 * (1.0 / (1.0 + total_distance))
    return branches, total_distance, branch_distances, fitness, coverage

target_functions = {
    "real": target_function_real,
    "complex": target_function_complex
}
# Usaremos la función compleja para este ejemplo multiobjetivo.
TARGET_FUNCTION = target_functions["complex"]

# ========================================================
# 3. Clase Chromosome: Representación y Operadores del GA
# ========================================================
class Chromosome:
    template = template
    def __init__(self, genes=None):
        if genes is not None:
            self.genes = genes
        else:
            self.genes = self._random_init(Chromosome.template)
        self.fitness = None
        self.covered_branches = set()
        self.branch_distance = None
        self.coverage = None
    def _random_init(self, schema):
        t = schema.get("type")
        if t == "int":
            return random.randint(schema.get("min", 0), schema.get("max", 100))
        elif t == "float":
            return random.uniform(schema.get("min", 0.0), schema.get("max", 1.0))
        elif t == "choice":
            options = schema.get("options", [])
            return random.choice(options) if options else None
        elif t == "string":
            length = schema.get("length", 5)
            allowed_chars = schema.get("chars", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            return ''.join(random.choice(allowed_chars) for _ in range(length))
        elif t == "nested":
            result = {}
            sub_schema = schema.get("schema", {})
            for key, sub in sub_schema.items():
                result[key] = self._random_init(sub)
            return result
        elif t == "list":
            length = schema.get("length", 0)
            elem_schema = schema.get("element")
            return [self._random_init(elem_schema) for _ in range(length)]
        else:
            if isinstance(schema, dict):
                result = {}
                for key, sub in schema.items():
                    result[key] = self._random_init(sub)
                return result
            return None
    def mutate(self, mutation_rate):
        def mutate_value(val, schema):
            t = schema.get("type")
            if random.random() >= mutation_rate:
                return val
            if t == "int":
                new_val = val + (1 if random.random() < 0.5 else -1)
                return max(schema.get("min", 0), min(new_val, schema.get("max", 100)))
            elif t == "float":
                range_span = schema.get("max", 1.0) - schema.get("min", 0.0)
                delta = random.uniform(-0.1 * range_span, 0.1 * range_span)
                new_val = val + delta
                return max(schema.get("min", 0.0), min(new_val, schema.get("max", 1.0)))
            elif t == "choice":
                options = schema.get("options", [])
                if len(options) > 1:
                    choices = [opt for opt in options if opt != val]
                    return random.choice(choices)
                return val
            elif t == "string":
                if not val:
                    return val
                pos = random.randrange(len(val))
                allowed_chars = schema.get("chars", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                new_char = random.choice(allowed_chars)
                if new_char == val[pos] and len(allowed_chars) > 1:
                    alt_chars = [ch for ch in allowed_chars if ch != val[pos]]
                    new_char = random.choice(alt_chars)
                new_value = list(val)
                new_value[pos] = new_char
                return ''.join(new_value)
            elif t == "nested":
                sub_schema = schema.get("schema", {})
                new_nested = {}
                for k, sub in sub_schema.items():
                    new_nested[k] = mutate_value(val[k], sub)
                return new_nested
            elif t == "list":
                new_list = list(val)
                length = schema.get("length", len(new_list))
                elem_schema = schema.get("element")
                for i in range(min(length, len(new_list))):
                    new_list[i] = mutate_value(new_list[i], elem_schema)
                return new_list
            else:
                if isinstance(val, dict):
                    new_val = {}
                    for k, subval in val.items():
                        sub_schema = schema.get(k, {})
                        new_val[k] = mutate_value(subval, sub_schema)
                    return new_val
                return val
        top_schema = Chromosome.template
        if top_schema.get("type") == "nested" or "schema" in top_schema:
            schema = top_schema.get("schema", top_schema)
            for key, sub_schema in schema.items():
                self.genes[key] = mutate_value(self.genes[key], sub_schema)
        else:
            self.genes = mutate_value(self.genes, top_schema)
    @classmethod
    def crossover(cls, parent1, parent2):
        keys = list(cls.template.get("schema", cls.template).keys())
        if len(keys) <= 1:
            child1_genes = deepcopy(parent1.genes)
            child2_genes = deepcopy(parent2.genes)
        else:
            point = random.randint(1, len(keys) - 1)
            child1_genes = {}
            child2_genes = {}
            for i, key in enumerate(keys):
                if i < point:
                    child1_genes[key] = deepcopy(parent1.genes[key])
                    child2_genes[key] = deepcopy(parent2.genes[key])
                else:
                    child1_genes[key] = deepcopy(parent2.genes[key])
                    child2_genes[key] = deepcopy(parent1.genes[key])
        return Chromosome(child1_genes), Chromosome(child2_genes)
    def repair(self):
        def repair_value(val, schema):
            t = schema.get("type")
            if t == "int":
                if val < schema.get("min", 0):
                    val = schema.get("min", 0)
                if val > schema.get("max", 100):
                    val = schema.get("max", 100)
                return val
            elif t == "float":
                if val < schema.get("min", 0.0):
                    val = schema.get("min", 0.0)
                if val > schema.get("max", 1.0):
                    val = schema.get("max", 1.0)
                return val
            elif t == "choice":
                options = schema.get("options", [])
                if options and val not in options:
                    val = random.choice(options)
                return val
            elif t == "string":
                max_len = schema.get("max_length")
                if max_len and len(val) > max_len:
                    val = val[:max_len]
                allowed_chars = schema.get("chars")
                if allowed_chars:
                    val = ''.join(ch if ch in allowed_chars else random.choice(allowed_chars) for ch in val)
                return val
            elif t == "nested":
                sub_schema = schema.get("schema", {})
                if not isinstance(val, dict):
                    return self._random_init(schema)
                for k, sub in sub_schema.items():
                    if k in val:
                        val[k] = repair_value(val[k], sub)
                    else:
                        val[k] = self._random_init(sub)
                return val
            elif t == "list":
                desired_length = schema.get("length")
                elem_schema = schema.get("element")
                if desired_length is not None:
                    if len(val) > desired_length:
                        val = val[:desired_length]
                    elif len(val) < desired_length:
                        for _ in range(desired_length - len(val)):
                            val.append(self._random_init(elem_schema))
                for i, item in enumerate(val):
                    val[i] = repair_value(item, elem_schema)
                return val
            else:
                if isinstance(val, dict):
                    for k, subval in val.items():
                        if k in schema:
                            val[k] = repair_value(subval, schema[k])
                    return val
                return val
        top_schema = Chromosome.template
        if top_schema.get("type") == "nested" or "schema" in top_schema:
            schema = top_schema.get("schema", top_schema)
            for key, sub_schema in schema.items():
                if key in self.genes:
                    self.genes[key] = repair_value(self.genes[key], sub_schema)
                else:
                    self.genes[key] = self._random_init(sub_schema)
        else:
            self.genes = repair_value(self.genes, top_schema)
    def distance_to(self, other):
        def gene_distance(val1, val2, schema):
            t = schema.get("type")
            if t in ["int", "float"]:
                mn = schema.get("min", 0)
                mx = schema.get("max", 1)
                if mx == mn:
                    return 0
                return abs(val1 - val2) / (mx - mn)
            elif t in ["choice", "string"]:
                return 0 if val1 == val2 else 1
            elif t == "nested":
                sub_schema = schema.get("schema", {})
                total = 0
                count = 0
                for key, sub in sub_schema.items():
                    total += gene_distance(val1.get(key), val2.get(key), sub)
                    count += 1
                return total / count if count > 0 else 0
            elif t == "list":
                elem_schema = schema.get("element")
                dists = [gene_distance(v1, v2, elem_schema) for v1, v2 in zip(val1, val2)]
                return sum(dists) / len(dists) if dists else 0
            else:
                return 0
        dist = 0
        schema = Chromosome.template.get("schema", Chromosome.template)
        keys = list(schema.keys())
        for key in keys:
            dist += gene_distance(self.genes.get(key), other.genes.get(key), schema[key])
        return dist / len(keys) if keys else 0
    def evaluate_fitness(self):
        result = TARGET_FUNCTION(self)  # (branches, total_distance, branch_distances, fitness, coverage)
        self.covered_branches = result[0]
        self.branch_distance = result[1]
        self.fitness = result[3]
        self.coverage = result[4]
        return self.fitness

# ========================================================
# 4. Función para actualizar la fitness con diversidad
# ========================================================
def update_fitness_with_diversity(population, alpha=0.8):
    N = len(population)
    avg_diversities = []
    for ind in population:
        total_dist = 0
        for other in population:
            if ind is not other:
                total_dist += ind.distance_to(other)
        avg_div = total_dist / (N - 1) if N > 1 else 0
        avg_diversities.append(avg_div)
        ind.avg_diversity = avg_div
    max_div = max(avg_diversities) if avg_diversities else 1
    for ind in population:
        norm_div = ind.avg_diversity / max_div if max_div > 0 else 0
        ind.composite_fitness = alpha * ind.fitness + (1 - alpha) * norm_div
    for ind in population:
        ind.fitness = ind.composite_fitness

# ========================================================
# 5. Función para calcular la diversidad global
# ========================================================
def calculate_population_diversity(population):
    n = len(population)
    if n < 2:
        return 0
    total_dist = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total_dist += population[i].distance_to(population[j])
            count += 1
    return total_dist / count if count > 0 else 0

# ========================================================
# 6. Función tournament_selection
# ========================================================
def tournament_selection(population, k=3):
    selected = random.sample(population, k)
    selected.sort(key=lambda x: x.fitness, reverse=True)
    return selected[0]

# ========================================================
# 7. Algoritmo Genético Principal con ajuste dinámico de mutación
# ========================================================
def genetic_algorithm(pop_size, generations, crossover_rate, base_mutation_rate):
    population = [Chromosome() for _ in range(pop_size)]
    for individual in population:
        individual.repair()
        individual.evaluate_fitness()
    best_fitness_history = []
    avg_fitness_history = []
    diversity_history = []
    mutation_rate = base_mutation_rate
    diversity_threshold = 0.15
    for gen in range(generations):
        diversity = calculate_population_diversity(population)
        diversity_history.append(diversity)
        if diversity < diversity_threshold:
            mutation_rate = min(base_mutation_rate * 1.5, 0.5)
        else:
            mutation_rate = base_mutation_rate
        update_fitness_with_diversity(population, alpha=0.8)
        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            if random.random() < crossover_rate:
                child1, child2 = Chromosome.crossover(parent1, parent2)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            child1.repair()
            child2.repair()
            child1.evaluate_fitness()
            child2.evaluate_fitness()
            new_population.extend([child1, child2])
        population.extend(new_population)
        population.sort(key=lambda x: x.fitness, reverse=True)
        population = population[:pop_size]
        update_fitness_with_diversity(population, alpha=0.8)
        best_fitness = population[0].fitness
        avg_fitness = sum(ind.fitness for ind in population) / pop_size
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        if (gen+1) % 10 == 0 or gen == 0:
            print(f"Generación {gen+1}: Mejor fitness = {best_fitness:.4f}, Promedio = {avg_fitness:.4f}, Diversidad = {diversity:.4f}, MutRate = {mutation_rate:.3f}")
    return population, best_fitness_history, avg_fitness_history, diversity_history

# ========================================================
# 8. Función para graficar la evolución
# ========================================================
def plot_evolution(best_history, avg_history, diversity_history):
    generations = range(1, len(best_history)+1)
    plt.figure(figsize=(14, 6))
    plt.subplot(1,2,1)
    plt.plot(generations, best_history, label='Mejor Fitness', marker='o')
    plt.plot(generations, avg_history, label='Fitness Promedio', marker='x')
    plt.xlabel('Generación')
    plt.ylabel('Composite Fitness')
    plt.title('Evolución del Fitness')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(generations, diversity_history, color='purple', marker='s')
    plt.xlabel('Generación')
    plt.ylabel('Diversidad Global')
    plt.title('Diversidad de la Población')
    plt.tight_layout()
    plt.show()

# ========================================================
# 9. Bloque Principal: Parámetros e interfaz de usuario
# ========================================================
if __name__ == '__main__':
    print("Seleccione la función objetivo a evaluar:")
    print("real    : Evaluar función real 'example' (simple)")
    print("complex : Evaluar función compleja 'complex'")
    choice = input("Ingrese 'real' o 'complex': ").strip()
    if choice not in target_functions:
        print("Selección inválida. Se usará la función 'complex' por defecto.")
        choice = "complex"
    TARGET_FUNCTION = target_functions[choice]
    
    try:
        pop_size = int(input("Ingrese el tamaño de la población: "))
    except:
        pop_size = 100
        print("Valor no válido, se usará 100.")
    try:
        generations = int(input("Ingrese el número de generaciones: "))
    except:
        generations = 100
        print("Valor no válido, se usará 100.")
    try:
        crossover_rate = float(input("Ingrese la tasa de cruce (ej. 0.8): "))
    except:
        crossover_rate = 0.8
        print("Valor no válido, se usará 0.8.")
    try:
        base_mutation_rate = float(input("Ingrese la tasa base de mutación (ej. 0.15): "))
    except:
        base_mutation_rate = 0.15
        print("Valor no válido, se usará 0.15.")
    
    population, best_history, avg_history, diversity_history = genetic_algorithm(
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        base_mutation_rate=base_mutation_rate
    )
    
    best_individual = population[0]
    print("\nMejor solución encontrada:")
    print(best_individual.genes)
    print("Ramas cubiertas por la mejor solución:", best_individual.covered_branches)
    if choice == "complex":
        print(f"Coverage: {best_individual.coverage*100:.1f}%")
        print(f"Total de ramas (ideal: 6): {len(best_individual.covered_branches)}")
    else:
        print(f"Total de ramas (ideal: 3): {len(best_individual.covered_branches)}")
    print()
    
    total_branches = set()
    for ind in population:
        total_branches.update(ind.covered_branches)
    if choice == "complex":
        print("Ramas cubiertas en la población final:", total_branches)
        print(f"Total de ramas cubiertas en la población: {len(total_branches)} (ideal: 6)")
    else:
        print("Ramas cubiertas en la población final:", total_branches)
        print(f"Total de ramas cubiertas en la población: {len(total_branches)} (ideal: 3)")
    print()
    
    print("Población final de datos de prueba:")
    for i, ind in enumerate(population):
        print(f"Test {i+1}: {ind.genes} | Fitness: {ind.fitness:.4f} | Coverage: {ind.coverage*100:.1f}%")
    
    plot_evolution(best_history, avg_history, diversity_history)
