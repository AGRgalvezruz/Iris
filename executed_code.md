
# Iris GP

## Main program


```python
import random
import operator
import csv
import itertools

import numpy

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
pset.addEphemeralConstant("rand8", lambda: random.random() * 8, float)
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
    

    #----------Iris-versicolor vs. Iris-virginica----------
    pop_2 = toolbox.population(n=500)
    hof_2 = tools.HallOfFame(1)

    stats_2 = tools.Statistics(lambda ind: ind.fitness.values)
    stats_2.register("avg", numpy.mean)
    stats_2.register("std", numpy.std)
    stats_2.register("min", numpy.min)
    stats_2.register("max", numpy.max)
    
    toolbox.register("evaluate", evalIrisbase, matrix=irisNoSetosa)
    
    print("")
    print(str("Iris-versicolor vs. Iris-virginica:"))
    algorithms.eaSimple(pop_2, toolbox, 0.5, 0.2, 100, stats_2, halloffame=hof_2)
    
    return pop_1, stats_1, hof_1, pop_2, stats_2, hof_2,
```


```python
pop_1, stats_1, hof_1, pop_2, stats_2, hof_2 = main()
```

    Iris-setosa vs. Iris-versicolor, Iris-virginica:
    gen	nevals	avg   	std    	min	max
    0  	500   	74.342	29.7554	0  	150
    1  	286   	91.182	28.6245	0  	150
    2  	310   	95.606	28.9121	0  	150
    3  	290   	101.812	31.1742	0  	150
    4  	318   	103.284	35.3048	0  	150
    5  	312   	112.474	36.0421	0  	150
    6  	304   	118.014	38.1837	0  	150
    7  	294   	120.92 	38.0604	0  	150
    8  	289   	124.746	35.8722	0  	150
    9  	284   	122.818	36.9316	0  	150
    10 	324   	122.476	36.7797	0  	150
    11 	287   	126.624	33.6986	0  	150
    12 	301   	124.554	36.6292	0  	150
    13 	296   	126.644	34.4824	0  	150
    14 	299   	124.484	37.7413	0  	150
    15 	298   	126.526	33.6616	0  	150
    16 	327   	123.174	37.4669	0  	150
    17 	296   	127.656	34.7889	0  	150
    18 	311   	126.702	35.2982	0  	150
    19 	314   	129.126	34.7612	0  	150
    20 	324   	130.024	32.1293	50 	150
    21 	294   	134.76 	29.7944	23 	150
    22 	300   	131.902	33.9628	0  	150
    23 	309   	132.812	31.9716	0  	150
    24 	302   	134.638	30.1015	0  	150
    25 	292   	133.412	30.8291	0  	150
    26 	309   	132.114	32.3771	0  	150
    27 	292   	133.132	31.6448	0  	150
    28 	299   	132.78 	32.5358	0  	150
    29 	294   	134.9  	28.5051	46 	150
    30 	284   	137.276	29.077 	0  	150
    31 	314   	135.056	29.7378	0  	150
    32 	275   	137.352	27.7188	50 	150
    33 	293   	137.958	26.6163	0  	150
    34 	300   	137.978	27.2155	0  	150
    35 	294   	136.94 	28.2682	0  	150
    36 	294   	138.142	27.8978	26 	150
    37 	297   	138.676	26.4331	0  	150
    38 	293   	139.858	24.8588	50 	150
    39 	303   	138.438	26.7795	25 	150
    40 	309   	136.49 	28.7439	49 	150
    41 	326   	137.924	27.0834	0  	150
    42 	304   	140.654	23.5595	50 	150
    43 	287   	141.94 	21.0292	50 	150
    44 	299   	139.076	26.7892	0  	150
    45 	296   	139.69 	25.2784	50 	150
    46 	312   	142.468	21.7173	50 	150
    47 	288   	142.668	21.3723	50 	150
    48 	286   	142.052	21.478 	0  	150
    49 	313   	140.238	25.268 	0  	150
    50 	306   	140.586	24.651 	5  	150
    51 	287   	140.31 	25.3635	0  	150
    52 	289   	142.04 	23.0335	0  	150
    53 	313   	142.43 	21.8461	46 	150
    54 	279   	144.254	20.183 	50 	150
    55 	315   	143.55 	19.4312	0  	150
    56 	283   	142.108	23.0014	50 	150
    57 	308   	142.32 	23.1744	6  	150
    58 	274   	142.836	21.5321	0  	150
    59 	298   	141.548	24.5359	0  	150
    60 	294   	143.174	21.2868	50 	150
    61 	304   	143.21 	19.8997	50 	150
    62 	293   	143.132	21.3394	48 	150
    63 	297   	141.528	23.5617	50 	150
    64 	317   	144.554	19.4743	50 	150
    65 	294   	143.632	21.244 	0  	150
    66 	285   	144.962	18.8638	0  	150
    67 	295   	144.49 	18.5658	50 	150
    68 	300   	144.046	20.2476	0  	150
    69 	320   	144.094	19.0537	50 	150
    70 	313   	142.828	23.3055	0  	150
    71 	304   	143.454	22.2107	0  	150
    72 	294   	143.594	21.1798	50 	150
    73 	304   	144.58 	18.2243	50 	150
    74 	307   	143.958	20.0274	50 	150
    75 	306   	144.878	17.5024	50 	150
    76 	296   	143.7  	22.3466	0  	150
    77 	262   	144.808	19.8059	0  	150
    78 	296   	144.368	20.2258	0  	150
    79 	313   	144.186	19.674 	0  	150
    80 	293   	143.474	20.8227	5  	150
    81 	319   	143.47 	20.5398	50 	150
    82 	287   	145.692	18.4005	0  	150
    83 	296   	144.6  	19.0366	50 	150
    84 	296   	145.314	17.5936	50 	150
    85 	306   	144.83 	19.6594	50 	150
    86 	288   	144.228	20.8347	0  	150
    87 	293   	145.29 	19.33  	0  	150
    88 	304   	145.276	19.0375	50 	150
    89 	311   	145.156	18.7232	50 	150
    90 	316   	145.416	17.4518	50 	150
    91 	305   	145.9  	18.3899	0  	150
    92 	289   	145.12 	18.6626	0  	150
    93 	304   	144.584	19.724 	50 	150
    94 	308   	146.212	16.1902	6  	150
    95 	303   	145.876	17.7547	50 	150
    96 	331   	144.628	19.9822	32 	150
    97 	298   	146.508	15.8505	50 	150
    98 	298   	145.33 	18.2142	39 	150
    99 	302   	144.844	18.9732	50 	150
    100	297   	144.63 	19.2116	50 	150
    
    Iris-versicolor vs. Iris-virginica:
    gen	nevals	avg   	std    	min	max
    0  	500   	50.194	4.09297	27 	94 
    1  	306   	51.29 	6.12225	10 	94 
    2  	300   	52.194	7.76456	10 	94 
    3  	294   	53.478	9.52562	6  	94 
    4  	309   	55.742	11.464 	8  	94 
    5  	317   	57.356	12.3934	6  	94 
    6  	296   	59.082	13.1566	8  	94 
    7  	303   	61.504	14.3549	27 	94 
    8  	290   	63.404	15.3041	33 	94 
    9  	314   	64.08 	17.215 	37 	94 
    10 	308   	64.824	18.6888	6  	94 
    11 	300   	67.34 	20.1724	6  	94 
    12 	298   	69.908	21.2104	6  	94 
    13 	352   	68.656	21.9824	6  	94 
    14 	317   	69.37 	22.5052	6  	94 
    15 	288   	73.448	22.2435	6  	94 
    16 	283   	73.752	22.9217	6  	94 
    17 	295   	73.292	23.0709	6  	94 
    18 	311   	72.998	22.3858	6  	94 
    19 	302   	73.392	22.7766	6  	94 
    20 	305   	74.186	22.6382	6  	94 
    21 	291   	76.014	22.4692	6  	94 
    22 	322   	72.862	22.3005	6  	94 
    23 	323   	73.544	22.4677	6  	94 
    24 	309   	74.618	22.6671	6  	94 
    25 	285   	75.69 	21.6892	44 	94 
    26 	308   	76.358	21.8439	6  	94 
    27 	296   	76.5  	22.2444	6  	94 
    28 	278   	75.498	21.9231	6  	94 
    29 	295   	75.51 	21.6693	49 	94 
    30 	316   	71.632	22.9018	6  	94 
    31 	291   	73.528	22.4543	6  	94 
    32 	283   	79.076	21.659 	6  	94 
    33 	303   	77.834	21.485 	6  	94 
    34 	286   	78.544	20.9201	50 	94 
    35 	316   	78.19 	21.1391	6  	94 
    36 	290   	78.878	21.1655	6  	94 
    37 	286   	79.75 	20.6866	6  	94 
    38 	309   	78.928	20.8029	44 	94 
    39 	297   	82.016	19.5393	22 	94 
    40 	307   	81.044	20.2992	6  	94 
    41 	292   	84.244	18.2508	44 	94 
    42 	304   	82.628	19.1138	48 	94 
    43 	306   	85.41 	17.6813	6  	94 
    44 	304   	86.81 	16.0476	50 	94 
    45 	320   	86.614	16.3639	42 	94 
    46 	287   	88.764	14.1851	47 	94 
    47 	309   	87.656	15.4452	32 	94 
    48 	325   	86.654	16.3561	50 	94 
    49 	277   	88.656	14.3177	44 	94 
    50 	298   	87.264	15.7327	50 	94 
    51 	303   	88.54 	14.4138	50 	94 
    52 	306   	89.082	13.8301	49 	94 
    53 	277   	87.964	15.0528	50 	94 
    54 	299   	88.018	14.9576	48 	94 
    55 	306   	87.458	15.7398	6  	94 
    56 	315   	89.44 	13.3518	50 	94 
    57 	322   	88.362	14.6198	41 	94 
    58 	305   	88.484	14.4833	50 	94 
    59 	296   	89.226	13.5398	50 	95 
    60 	289   	89.15 	13.7097	50 	95 
    61 	313   	87.72 	15.3777	44 	95 
    62 	320   	88.114	14.9339	34 	95 
    63 	282   	90.046	12.5097	50 	95 
    64 	292   	89.142	13.8986	50 	95 
    65 	315   	89.204	13.8724	50 	95 
    66 	317   	88.25 	15.033 	50 	95 
    67 	269   	90.454	12.7252	49 	95 
    68 	325   	89.022	14.8052	50 	95 
    69 	295   	90.064	13.2804	50 	95 
    70 	287   	89.764	13.9114	50 	95 
    71 	270   	90.324	13.2069	50 	95 
    72 	315   	90.288	13.1638	50 	95 
    73 	312   	90.91 	12.3463	50 	95 
    74 	299   	90.712	12.563 	49 	95 
    75 	316   	89.574	13.9018	49 	95 
    76 	313   	91.574	11.5617	50 	95 
    77 	320   	90.688	12.647 	50 	95 
    78 	290   	91.078	12.327 	50 	95 
    79 	294   	92.044	10.7667	50 	95 
    80 	313   	90.74 	12.856 	17 	95 
    81 	286   	91.626	11.1833	50 	96 
    82 	300   	91.41 	11.7121	50 	95 
    83 	304   	91.428	11.7849	48 	95 
    84 	328   	89.83 	13.9436	50 	95 
    85 	306   	91.418	11.8632	50 	95 
    86 	296   	90.716	12.6024	50 	95 
    87 	303   	90.696	12.82  	50 	95 
    88 	284   	92.704	9.46089	50 	95 
    89 	311   	91.176	12.1419	50 	96 
    90 	274   	92.264	10.4024	49 	96 
    91 	302   	92.664	9.40995	50 	96 
    92 	325   	92.076	10.6804	50 	96 
    93 	324   	92.102	10.7776	50 	96 
    94 	300   	91.494	11.7427	50 	96 
    95 	286   	92.392	10.52  	49 	96 
    96 	288   	92.526	10.4106	50 	96 
    97 	313   	93.17 	9.58087	50 	96 
    98 	307   	92.84 	10.6985	44 	96 
    99 	317   	92.8  	11.0263	50 	96 
    100	302   	92.842	10.8669	50 	96 



```python
print("Hall of fame IT_1:")
print(str(hof_1[0])) #print the best individual which has the best fitness
print("")
print("Hall of fame IT_2:")
print(str(hof_2[0]))
```

    Hall of fame IT_1:
    gt(sw, pl)
    
    Hall of fame IT_2:
    and_(or_(and_(le(pw, 1.709820965411212), lt(pl, 5.031718503391735)), gt(pw, sw)), le(pw, 7.911868533317178))


## Function to draw decision trees


```python
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

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
```


```python
plotting(hof_1[0])
plotting(hof_2[0])
```

![se_vs_ve-vi.png](attachment:se_vs_ve-vi.png)
![ve_vs_vi.png](attachment:ve_vs_vi.png)
