# Store all functions use in MinLa algorithm by Eduardo Rodriguez-Tello et al.
# https://www.sciencedirect.com/science/article/pii/S0305054807000676

import MinLA_functions as MinLA
from tulip import tlp
import numpy

graph = MinLA.main('AMAZ_step3_V6_100k.csv.tlp', None)

# graph = MinLA.fim(graph, False) # initial placement

# best_solution = MinLA.tssa_objective_function(graph) # A REVOIR

#MinLA_functions.Get_viewLayout(graph)

# tlp.saveGraph(graph, 'test_V10_done.tlp')

# position = MinLA.get_position(graph)

# C_inf, Sig_inf = MinLA.generate_independant_solutions(graph, position, 10**3)

# output = MinLA.annealing(
#     MinLA.fim(graph, False),
#     MinLA.cost_function,
#     MinLA.random_node,
#     MinLA.temperature(graph, position, C_inf, Sig_inf, best_solution),
#     maxsteps=10000,
#     distance=0.1,
#     debug=True)

# print(graph.nodes())

output = MinLA.minLA_each_components(graph, type_step='proportion', nb_steps=20, ancient=False)

# print("Mean Cost funct")
# print(C_inf)
# print(Sig_inf)
# print('Gamma')

# print(MinLA.gamma_inf(graph, C_inf, Sig_inf))

tlp.saveGraph(output, 'test_V9_done.tlp')

