# Store all functions use in MinLa algorithm by Eduardo Rodriguez-Tello et al.
# https://www.sciencedirect.com/science/article/pii/S0305054807000676

from tulip import tlp
import math
import random
import numpy as np
import scipy.stats as st
import numpy.random as rn
import copy

# get_key(position, node_1)**2/1000
# A CHANGER

def random_node(graph):
    return graph.getRandomNode()

def get_key(dict, val):
    for key, value in dict.items():
        if val == value:
            return key

    return "key doesn't exist"

def main(file, graph):
    graph = tlp.loadGraph(file, graph)
    ELP = graph.getDoubleProperty("ELP")
    Weigth = graph.getDoubleProperty("Weigth")
    Voisin = graph.getIntegerProperty("Voisin")
    Classe = graph.getStringProperty("Classe")
    viewBorderColor = graph.getColorProperty("viewBorderColor")
    viewBorderWidth = graph.getDoubleProperty("viewBorderWidth")
    viewColor = graph.getColorProperty("viewColor")
    viewFont = graph.getStringProperty("viewFont")
    viewFontSize = graph.getIntegerProperty("viewFontSize")
    viewIcon = graph.getStringProperty("viewIcon")
    viewLabel = graph.getStringProperty("viewLabel")
    viewLabelBorderColor = graph.getColorProperty("viewLabelBorderColor")
    viewLabelBorderWidth = graph.getDoubleProperty("viewLabelBorderWidth")
    viewLabelColor = graph.getColorProperty("viewLabelColor")
    viewLabelPosition = graph.getIntegerProperty("viewLabelPosition")
    viewLayout = graph.getLayoutProperty("viewLayout")
    viewMetric = graph.getDoubleProperty("viewMetric")
    viewRotation = graph.getDoubleProperty("viewRotation")
    viewSelection = graph.getBooleanProperty("viewSelection")
    viewShape = graph.getIntegerProperty("viewShape")
    viewSize = graph.getSizeProperty("viewSize")
    viewSrcAnchorShape = graph.getIntegerProperty("viewSrcAnchorShape")
    viewSrcAnchorSize = graph.getSizeProperty("viewSrcAnchorSize")
    viewTexture = graph.getStringProperty("viewTexture")
    viewTgtAnchorShape = graph.getIntegerProperty("viewTgtAnchorShape")
    viewTgtAnchorSize = graph.getSizeProperty("viewTgtAnchorSize")

    for e in graph.getEdges():
        Weigth[e] = viewMetric[e]
        ELP[e] = (1 - viewMetric[e])
        # print(graph.getEdgePropertiesValues(e))
        # print(ELP[e])
        if viewColor[graph.source(e)] == viewColor[graph.target(e)]:
            viewColor[e] = viewColor[graph.source(e)]
        # print(viewColor[graph.source(e)])
        # print(viewColor[graph.target(e)])

    for n in graph.getNodes():
        print(n)
        vois = 0
        if viewColor[n] == (200, 0, 0, 255):
            Classe[n] = 'frontier'
        elif viewColor[n] == (0, 200, 0, 255):
            Classe[n] = '1'
        else:
            Classe[n] = '0'
        for i in graph.getInOutEdges(n):
            vois += 1
        Voisin[n] = vois
        # print(viewColor[n])
        # print(ELP[e])

    return graph

def fim(graph, less_degree = False):

    name = graph.getStringProperty("viewLabel")
    viewLayout = graph.getLayoutProperty("viewLayout")
    U = graph.getBooleanProperty('U')
    F = graph.getBooleanProperty('F')
    P = graph.getBooleanProperty('F')
    Tl = graph.getIntegerProperty('Tl')
    Tr = graph.getIntegerProperty('Tr')
    Dg = graph.getIntegerProperty("Voisin")
    Sf = graph.getIntegerProperty("Sf")
    for n in graph.getNodes():
        P[n] = False
        U[n] = True

    nb_nodes = graph.numberOfNodes()
    nb_nodes_done = []

    # Initial node is the one with the less degree or a random_one
    node = graph.getRandomNode()
    if less_degree:
        for a_node in graph.getNodes():
            if Dg[a_node] < Dg[node]:
                node = a_node

    # Place every node
    while len(nb_nodes_done) < nb_nodes:
        viewLayout[node] = (len(nb_nodes_done)*1, len(nb_nodes_done)**2/18, 0)
        nb_nodes_done.append(name[node])
        P[node] = True
        U[node] = False
        for a_node in graph.getNodes():
            Tl[a_node] = 0
            for neig in graph.getInOutNodes(a_node):
                if P[neig]:
                    Tl[a_node] += 1
            Tr[a_node] = Dg[a_node] - Tl[a_node]
            Sf[a_node] = Tr[a_node] - Tl[a_node]

            F[neig] = True
            if name[a_node] not in nb_nodes_done:
                node = a_node
        for min_node in graph.getNodes():
            if Sf[node] >= Sf[min_node] and U[min_node]:
                node = min_node

    return graph

def get_viewLayout(graph):
    viewLayout = graph.getLayoutProperty("viewLayout")
    position = get_position(graph)
    # print(position)
    p = 0.9
    for n in graph.getNodes():
        graph, position = move_node_n3(graph, position, n, p)
        position = get_position(graph)
    return graph

def get_position(graph):
    viewLayout = graph.getLayoutProperty("viewLayout")
    position = dict()
    for n in graph.getNodes():
        position[int(viewLayout[n][0])] = n
    # print(position)
    return position

def set_position(graph, position):
    viewLayout = graph.getLayoutProperty("viewLayout")
    for n in graph.getNodes():
        viewLayout[n] = (get_key(position, n) * 1, get_key(position, n)**2/1000, 0)
    # print(position)
    return graph, position

# function from https://perso.crans.org/besson/publis/notebooks/Simulated_annealing_in_Python.html
def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        # print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        # print("    - Acceptance probabilty = {:.3g}...".format(p))
        return p

# function from https://perso.crans.org/besson/publis/notebooks/Simulated_annealing_in_Python.html
def annealing(graph, cost_function, random_neighbour, temperature, maxsteps=1000, distance=.1, p=.9, debug=True):
    """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
    state = graph
    cost = cost_function(state)
    states, costs = [get_position(graph)], [cost]
    tlp.saveGraph(state, './BSF.tlp')
    T = temperature
    sigma_temp = 1 # A CHANGER !!!
    for step in range(maxsteps):
        state = tlp.loadGraph('./BSF.tlp')
        fraction = step / float(maxsteps)
        # sigma_temp = np.std(costs)
        position = get_position(state)
        node = random_node(state)
        new_state, new_position, type = move_node_n3(state, position, node, p)
        new_cost = cost_function(new_state)
        if debug: print("Step #{:>2}/{:>2} : T = {:>4.3g}, cost = {:>4.3g}, new_cost = {:>4.3g}, type {} ...".format(step, maxsteps, T, cost, new_cost, type))
        if acceptance_probability(cost, new_cost, T) > rn.random():
            tlp.saveGraph(state, './BSF.tlp')
            state, cost = new_state, new_cost
            states.append(position)
            costs.append(cost)
            print("  ==> Accept it!")
        else:
            print("  ==> Reject it...")
        sigma_temp = 1 # A REVOIR !!!!
        T = new_temperature(T, sigma_temp, distance)
    return state, cost_function(state), states, costs

def tssa_objective_function(graph):
    """ Function to minimize."""
    viewLayout = graph.getLayoutProperty("viewLayout")
    differences = dict()
    for node in graph.getNodes():
        for neig in graph.getInOutNodes(node):
            if abs(int(viewLayout[node][0] - viewLayout[neig][0])) in differences: # A optimiser
                differences[abs(int(viewLayout[node][0] - viewLayout[neig][0]))] += 0.5
            else:
                differences[abs(int(viewLayout[node][0] - viewLayout[neig][0]))] = 0.5

    """LA = 0
    for key, value in differences.items():
        LA += key*value

    n = len(differences)
    tssa = 0
    for key, value in differences.items():
        tssa += (math.factorial(n)*value)/math.factorial(n+key)
    tssa += LA"""

    n = len(differences)
    tssa = 0
    for key, value in differences.items():
        tssa += key*value + (math.factorial(n) * int(value)) / math.factorial(n + int(key))
    # print(differences)
    # print(tssa)
    return tssa

def cost_function(graph):
    """ Cost of x = f(x)."""
    return tssa_objective_function(graph)

def swap_nodes(graph, position, node_1, node_2):
    viewLayout = graph.getLayoutProperty("viewLayout")
    provisoire = get_key(position, node_2)
    viewLayout[node_2] = (get_key(position, node_1) * 1, get_key(position, node_1)**2/1000, 0)
    position[get_key(position, node_1)] = node_2
    position[provisoire] = node_1
    viewLayout[node_1] = (provisoire * 1, provisoire**2/18, 0)

    return graph, position

def move_node_n1(graph, position, node):
    viewLayout = graph.getLayoutProperty("viewLayout")
    neighbors = []
    # print('node à traité')
    # print(node)

    # Neighbors of a node
    for neig in graph.getInOutNodes(node):
        neighbors.append(int(viewLayout[neig][0]))
        neighbors.sort()
        # print(position)

    # Mediane of neighbors
    if len(position)%2 == 0:
        median = int((1/2)*(neighbors[int(len(neighbors)/2)-1]+neighbors[int(1+len(neighbors)/2)-1]))
    else:
        # print(int((len(position) + 1) / 2))
        median = int(neighbors[(int((len(neighbors)+1)/2))-1])

    # Mediane of neighbors closer nodes
    MU = []
    MU_tssa = []
    for indice in range(median-2, median+2):
        if indice >= 1:
            MU.append(position[indice-1])
            MU_tssa.append(tssa_objective_function(swap_nodes(graph, position, node, MU[-1])[0]))

    # Tab of indices of best tssa value
    indices = [i for i, x in enumerate(MU_tssa) if x == min(MU_tssa)]

    # Move nodes to get the best change possible
    graph, position = swap_nodes(graph, position, node, position[random.choice(indices)])
    # print(MU_tssa[indices[0]])

    return graph, position

def move_node_n2(graph, position, node):
    viewLayout = graph.getLayoutProperty("viewLayout")
    neighbors = []
    # print('node à traité')
    # print(node)

    # Neighbors of a node
    for neig in graph.getInOutNodes(node):
        neighbors.append(int(viewLayout[neig][0]))
        neighbors.sort()
        # print(position)

    # ALL
    ALL = []
    ALL_tssa = []
    for indice in range(len(position)):
        if indice != get_key(position, node):
            ALL.append(position[indice])
            ALL_tssa.append(tssa_objective_function(swap_nodes(graph, position, node, ALL[-1])[0]))

    # Tab of indices of best tssa value
    indices = [i for i, x in enumerate(ALL_tssa) if x == min(ALL_tssa)]

    # Move nodes to get the best change possible
    graph, position = swap_nodes(graph, position, node, position[random.choice(indices)])
    # print(ALL_tssa[indices[0]])

    return graph, position

def move_node_n3(graph, position, node, p):
    if p >= random.random():
        result = move_node_n1(graph, position, node)
        return result[0], result[1], 'N1'
    else:
        result = move_node_n2(graph, position, node)
        return result[0], result[1], 'N2'

def generate_independant_solutions(graph, position, number):
    tssa = []
    for i in range(number):
        random_position = dict(zip(position.keys(), random.sample(list(position.values()), len(position))))
        graph, random_position = set_position(graph, random_position)
        tssa.append(cost_function(graph))
    mean = np.mean(tssa)
    std = np.std(tssa)

    return mean, std

def c_k(C_inf, Sig_inf, Tk):
    return C_inf - (Sig_inf**2/Tk)

def sigma_k(Sig_inf):
    return Sig_inf

def temperature(graph, position, C_inf, Sig_inf, best_solution):
    return Sig_inf**2/(C_inf - best_solution - gamma_inf(graph, C_inf, Sig_inf)*Sig_inf)

def gamma_inf(graph, C_inf, Sig_inf):
    r = 50000 # VALEUR DE R A VOIR !!!
    # Le reste a l'air juste
    p = 1 - abs(r)**(-1)
    p = 0.5 + p/2
    gamma_inf = st.norm.ppf(p, loc = C_inf, scale = Sig_inf)
    gamma_inf = (gamma_inf - C_inf)/Sig_inf
    return gamma_inf

def new_temperature(temp, sigma_temp, dist):
    return temp*(1+(math.log(1+dist)*temp)/(3*sigma_temp))**(-1)

#def stop(outputs, espilon):
