import random


def readFile(file_input):
    with open(file_input, "r") as f:
        array = []
        for line in f:
            array.append(line.rstrip())
    f.close()

    n = 0
    k = 0
    point_array = []

    for idx, val in enumerate(array):
        if idx == 0:  # get n, l
            array_value = val.split()
            n = int(array_value[0])
            k = int(array_value[1])
        else:
            point_array.insert(0, int(val))

    return n, k, point_array


class Population:
    def __init__(self, populationSize, isInit, genesSize):
        self.individuals = []
        self.populationSize = populationSize
        self.isInit = isInit
        self.genesSize = genesSize
        if isInit:
            for i in range(0, populationSize):
                # self.individuals[i] = Individual(genesSize).random()
                self.individuals.append(Individual(genesSize).random())

    def getFittest(self):
        result = Individual(self.individuals[0])
        for i in range(0, self.populationSize):
            if Individual(self.individuals[i]).getFitness() > result.getFiness():
                result = Individual(self.individuals[i])
        return result


class Individual:
    def __init__(self, genesSize):
        self.genes = []
        self.genesSize = genesSize

    def random(self):
        for i in range(0, self.genesSize):
            # self.genes[i] = (random.randint(0, 1) == 0)
            self.genes.append(random.randint(0, 1) == 0)
        return self.genes

    def getFitness(self):
        return FitnessCal.getFitness(self)


class GeneticAlgorithm:
    def __init__(self, uniform_rate, mutation_rate, tournament_size, elitism, population_size, loop, genes_size):
        self.uniform_rate = uniform_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.population_size = population_size
        self.loop = loop
        self.genes_size = genes_size

    def tournamentSelection(self, population):
        tournament = Population(self.tournament_size, False)
        for i in range(0, self.tournament_size):
            randomIdx = random.randint(self.population_size)
            tournament.individuals[i] = population.individuals[randomIdx]
        return tournament.getFittest()

    def crossOver(self, individual1, individual2):
        newIndividual = Individual()

        for i in range(0, self.genes_size):
            if random.random() <= self.mutation_rate:
                newIndividual.genes[i] = individual1.genes[i]
            else:
                newIndividual.genes[i] = individual2.genes[i]

        return newIndividual

    def mutate(self, individual):
        for i in range(0, len(individual.genes)):
            if random.random() == self.mutation_rate:
                if individual.genes[i]:
                    individual.genes[i] = False
                else:
                    individual.genes[i] = True
        return individual

    def evolutionPopulation(self, population):
        newPopulation = Population(self.population_size, False)
        initIdx = 0
        if self.elitism:
            newPopulation.individuals[0] = population.getFittest()
            initIdx = 1

        for i in range(initIdx, self.population_size):
            indi1 = self.tournamentSelection(population)
            indi2 = self.tournamentSelection(population)
            newIndi = self.crossOver(indi1, indi2)

            newIndi = self.mutate(newIndi)
            newPopulation.individuals[i] = newIndi

        return newPopulation

    def run(self):
        population = Population(self.population_size, True, self.genes_size)
        for i in range(0, self.loop):
            print(population.getFittest().getFitness())
            population = self.evolutionPopulation(population)
        return


class FitnessCal(object):
    k = 0
    n = 0
    weights = []

    @staticmethod
    def setK(k):
        FitnessCal.k = k

    @staticmethod
    def setN(n):
        FitnessCal.n = n

    @staticmethod
    def setWeights(weights):
        FitnessCal.weights = weights

    @staticmethod
    def getFitness(individual):
        finalScore = 0.0
        scorePer = []

        penalty = 0

        for i in range(0, FitnessCal.n):
            score = 0
            numberOne = 0
            for j in range(0, FitnessCal.n - 1):
                if individual.genes[j + (FitnessCal.n - 1) * i]:
                    if j < i:
                        score += FitnessCal.weights[j]
                    else:
                        score += FitnessCal.weights[j + 1]
                    numberOne += 1
            if FitnessCal.k > numberOne:
                # score += (f.k - numberOne) * -1000
                penalty = penalty + FitnessCal.k - numberOne
            else:
                # score += (numberOne - f.k) * -1000
                penalty = penalty + numberOne - FitnessCal.k
            scorePer[i] = (score * 1.0) / (numberOne * 1.0)

        if penalty > 1:
            finalScore = penalty * -1000.0
        else:
            for i in range(0, FitnessCal.n):
                for j in range(0, FitnessCal.n):
                    if i != j:
                        finalScore += abs(scorePer[j] - scorePer[i])
            finalScore = finalScore * -1.0 / (FitnessCal.n * (FitnessCal.n - 1))

        return finalScore


def main(file_input, file_output):
    # read input
    n, k, point_array = readFile(file_input)
    # run algorithm
    FitnessCal.setN(n)
    FitnessCal.setK(k)
    FitnessCal.setWeights(point_array)
    ga = GeneticAlgorithm(0.5, 0.01, 35, True, 500, 300, n * (n - 1))
    ga.run()

    # write output

    return


main('input.txt', 'output.txt')
