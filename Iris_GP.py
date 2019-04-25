import random
import operator
import csv
import itertools

import numpy

#---------------import for plotting------------------
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
#----------------------------------------------------

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import altergp as gp

# Read the iris list features and put it in a list of lists.
with open("iris.csv") as irisbase:
    irisReader = csv.reader(irisbase)
    irisDB = list(list(str(elem) if elem == "Iris-setosa" or elem == "Iris-versicolor" or elem == "Iris-virginica" else float(elem) for elem in row) for row in irisReader)

# Iris matrix: Iris-setosa = True; Iris-versicolor AND Iris-virginica = False
iris = list(list(bool(True) if elem == "Iris-setosa" else bool(False) if elem == "Iris-versicolor" or elem == "Iris-virginica" else float(elem) for elem in row) for row in irisDB)

# Iris matrix without class Iris-setosa
irisNoSetosa = list(list(bool(True) if elem == "Iris-versicolor" else bool(False) if elem == "Iris-virginica" else float(elem) for elem in row) for row in irisDB if row[4] != "Iris-setosa")

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 4), bool)
pset.renameArguments(ARG0="sl")
pset.renameArguments(ARG1="sw")
pset.renameArguments(ARG2="pl")
pset.renameArguments(ARG3="pw")

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)

# logic operators
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.le, [float, float], bool)
pset.addPrimitive(operator.ge, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(operator.ne, [float, float], bool)

# terminals
pset.addEphemeralConstant("rand80", lambda: random.random() * 8, float)
pset.addTerminal(True, bool)
pset.addTerminal(False, bool)

# Define the operator fitnessMax and create the individual who uses it
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalIrisbase(individual, matrix):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    result = sum(bool(func(*elem[:4])) is bool(elem[4]) for elem in matrix)

    return result,
    
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=6)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def plotting(individual):
	nodes, edges, labels = gp.graph(individual)
	g = nx.Graph()
	g.add_nodes_from(nodes)
	g.add_edges_from(edges)
	pos = graphviz_layout(g, prog="dot")

	nx.draw_networkx_nodes(g, pos)
	nx.draw_networkx_edges(g, pos)
	nx.draw_networkx_labels(g, pos, labels)
	plt.show()


def main():
    random.seed(20)

    #----------Iris-setosa vs. Iris-versicolor, Iris-virginica----------
    pop_1 = toolbox.population(n=500)
    hof_1 = tools.HallOfFame(1)

    stats_1 = tools.Statistics(lambda ind: ind.fitness.values)
    stats_1.register("avg", numpy.mean)
    stats_1.register("std", numpy.std)
    stats_1.register("min", numpy.min)
    stats_1.register("max", numpy.max)
    
    toolbox.register("evaluate", evalIrisbase, matrix=iris)

    print(str("Iris-setosa vs. Iris-versicolor, Iris-virginica:"))
    algorithms.eaSimple(pop_1, toolbox, 0.5, 0.2, 100, stats_1, halloffame=hof_1)
    
    print("")
    print("HallOfFame:")
    print (hof_1[0])
    print("")

    plotting(hof_1[0])

    #----------Iris-versicolor vs. Iris-virginica----------
    pop_2 = toolbox.population(n=500)
    hof_2 = tools.HallOfFame(1)

    stats_2 = tools.Statistics(lambda ind: ind.fitness.values)
    stats_2.register("avg", numpy.mean)
    stats_2.register("std", numpy.std)
    stats_2.register("min", numpy.min)
    stats_2.register("max", numpy.max)
    
    toolbox.register("evaluate", evalIrisbase, matrix=irisNoSetosa)

    print(str("Iris-versicolor vs. Iris-virginica:"))
    algorithms.eaSimple(pop_2, toolbox, 0.5, 0.2, 100, stats_2, halloffame=hof_2)
    
    print("")
    print("HallOfFame:")
    print (hof_2[0])
    print("")
    plotting(hof_2[0])
    return pop_1, stats_1, hof_1, pop_2, stats_2, hof_2,

if __name__ == "__main__":
    main()
