# Store all functions use in MinLa algorithm by Eduardo Rodriguez-Tello et al.
# https://www.sciencedirect.com/science/article/pii/S0305054807000676

from tulip import tlp
import math
import random
import numpy as np
import scipy.stats as st
import numpy.random as rn
import copy
import itertools

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
        # print(n)
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
    P = graph.getBooleanProperty('P')
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

            #F[neig] = True
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
    # print(state.nodes())
    tlp.saveGraph(state, './BSF.tlp')
    T = temperature
    sigma_temp = 1 # A CHANGER !!!
    for step in range(maxsteps):
        state = tlp.loadGraph('./BSF.tlp')
        if len(state.nodes()) !=0: # AAAA VOIIIR
            # print(state.nodes())
            fraction = step / float(maxsteps)
            # sigma_temp = np.std(costs)
            position = get_position(state)
            node = random_neighbour(state)
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
    if len(neighbors)%2 != 0:
        if len(neighbors)%2 == 0:
            # print(neighbors)
            median = int((1/2)*(neighbors[int(len(neighbors)/2)-1]+neighbors[int(1+len(neighbors)/2)-1]))
        else:
            # print(int((len(position) + 1) / 2))
            # print(neighbors)
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
    # print(indices)
    # Move nodes to get the best change possible
    if len(indices) !=0:
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

def minLA(graph, position_n, steps=500, ancient=True):
    viewLayout = graph.getLayoutProperty("viewLayout")
    classe = graph.getStringProperty("Classe")
    viewLabel = graph.getStringProperty("viewLabel")

    frontier_component = []
    for n in graph.nodes():
        if classe[n] == 'frontier':
            frontier_component.append(n)
    frontier_component = graph.inducedSubGraph(frontier_component, parentSubGraph=graph, name="comp_frontier")
    # print("len juste front : "+str(len(frontier_component.nodes())))
    if tlp.loadGraph('./comp/test_V9_'+str(graph.getName())+'_done.tlp') == None:
        # print("create fil")
        fimed = fim(frontier_component, False)  # initial placement
        tlp.saveGraph(fimed, './comp/test_V9_'+str(graph.getName())+'_done.tlp')
    fimed = tlp.loadGraph('./comp/test_V9_' + str(graph.getName()) + '_done.tlp')
    best_solution = tssa_objective_function(fimed)
    position = get_position(fimed)
    # print(len(position))
    # print(len(frontier_component.nodes()))
    C_inf, Sig_inf = generate_independant_solutions(fimed, position, 10 ** 3)

    if ancient:
        if tlp.loadGraph('./comp/test_V9_' + str(graph.getName()) + '_annealed_done.tlp') == None:
            output = annealing(
                fimed,
                cost_function,
                random_node,
                temperature(frontier_component, position, C_inf, Sig_inf, best_solution),
                maxsteps=steps,
                distance=0.1,
                debug=True)
            tlp.saveGraph(output[0], './comp/test_V9_' + str(graph.getName()) + '_annealed_done.tlp')
            del output
        new_graph = tlp.loadGraph('./comp/test_V9_' + str(graph.getName()) + '_annealed_done.tlp')
    else:
        output = annealing(
            fimed,
            cost_function,
            random_node,
            temperature(frontier_component, position, C_inf, Sig_inf, best_solution),
            maxsteps=steps,
            distance=0.1,
            debug=True)
        tlp.saveGraph(output[0], './comp/AMAZ/test_amaz_' + str(graph.getName()) + '_annealed_done.tlp')
        new_graph = tlp.newGraph()
        tlp.copyToGraph(new_graph, output[0])
        del output

    # no_front = []
    viewLayoutNew = new_graph.getLayoutProperty("viewLayout")
    viewLabelNew = new_graph.getStringProperty("viewLabel")
    for n in graph.nodes():
        if classe[n] == 'frontier':
            # same_node = viewLabelNew.getNodesEqualTo(viewLabel[n])
            for same_node in viewLabelNew.getNodesEqualTo(viewLabel[n]): # only one iter
                viewLayout[n] = viewLayoutNew[same_node]
            # viewLayout[n] = viewLayoutNew[same_node[0]]
            # print(viewLabelNew[same_node])
            # print(viewLabel[n])
            # no_front.append(n)
            # test +=1
        graph.addNode(n)
        # print(len(graph.nodes()))
    # print("len juste front graph nodes: " + str(test))
    tlp.saveGraph(graph, './comp/test_V9_' + str(graph.getName()) + '_annealed_allnodes_done.tlp')
    # A VOIR
    # print(frontier_component.nodes()+no_front)
    # final_graph = graph.inducedSubGraph(frontier_component.nodes()+no_front, parentSubGraph=graph, name="comp_frontier")
    final_graph = graph

    for n in final_graph.getNodes():
        viewLayout[n] = (viewLayout[n][0]+position_n,viewLayout[n][1],0)


    # final_graph = graph.inducedSubGraph(output.nodes(), parentSubGraph=graph, name="MinLA_final_graph")

    for n in final_graph.getNodes():
        viewLayout[n][1] += position_n
    position_n += len(frontier_component.nodes()) + 2
    return final_graph, position_n

def minLA_each_components(graph, type_step='fix', nb_steps=500, ancient=False):
    position_n = 0
    components = tlp.ConnectedTest.computeConnectedComponents(graph)
    # print(len(components))
    comps = []
    for i in range(len(components)):
        print(type(components[i]))
        print(type(graph))
        # pv_graph = tlp.newGraph()
        # print(components[i])
        # pv_graph = pv_graph.addNodes(components[i])

        comps.append(graph.inducedSubGraph(list(components[i]), parentSubGraph=graph, name="comp_"+str(i)))
        if len(comps[i].nodes()) != 0:
            if type_step == 'fix':
                comps[i], position_n = minLA(comps[i], position_n, nb_steps, ancient)
            elif type_step == 'proportion':
                comps[i], position_n = minLA(comps[i], position_n, (comps[i].numberOfNodes()*nb_steps), ancient)

    # print(comps)

    new_graph = tlp.newGraph()
    for i in range(len(comps)):
        print(i)
        tlp.copyToGraph(new_graph, comps[i])
        print(len(new_graph.nodes()))
        # new_graph = new_graph.inducedSubGraph(comps[i].nodes(), parentSubGraph=graph, name="minLA_Graph")
    return new_graph




#def stop(outputs, espilon):
