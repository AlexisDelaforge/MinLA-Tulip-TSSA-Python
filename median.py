import MinLA_functions as MinLA
import median_functions as median
from tulip import tlp
import numpy

# graph = MinLA.main('test_V9_done.tlp', None)
graph = tlp.loadGraph('/home/alexis/Project/MinLA/comp/test_V9_comp_4_annealed_allnodes_done.tlp', None)

# print(tulip.__version__)

graph = median.place_nodes(graph, 'median')
print(graph)
# tlp.saveGraph(graph, 'test_V9_done_mean.tlp')
tlp.saveGraph(graph, '/home/alexis/Project/MinLA/comp/test_provisoire.tlp')
