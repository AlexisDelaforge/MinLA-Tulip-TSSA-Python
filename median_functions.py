from tulip import tlp
import math
import numpy
import random
import numpy as np


def place_node(graph, done, node, type='median'):
    places = []
    viewLayout = graph.getLayoutProperty("viewLayout")
    distance = graph.getDoubleProperty("distance")
    classe = graph.getStringProperty("Classe")
    if classe[node] == 'frontier':
        viewLayout[node] = (distance[node], viewLayout[node][0], 0)
        done.append(node)
    else:
        for neig in graph.getInOutNodes(node):
            if neig in done:
                places.append(viewLayout[neig][1])
            else:
                print('not in for '+str(neig)+" neig of "+str(node))
        print(str(places) + " for " + str(node.id))
        if len(places) > 1:
            done.append(node)
            if type == 'median':
                y_node = np.median(places)
            else:
                y_node = np.mean(places)
        elif len(places) == 1:
            done.append(node)
            y_node = places[0]
        else:
            return graph, done
        print(y_node)
        x_node = distance[node]
        # print(viewLayout[node])
        # print(y_node)
        viewLayout[node] = (x_node, y_node, 0)
        print(viewLayout[node])
    return graph, done


def closest(graph, done):
    distance = graph.getDoubleProperty("distance")
    absolute = graph.getDoubleProperty("viewMetric")
    for n in graph.getNodes():
        absolute[n] = np.absolute(distance[n])
    for n in absolute.getSortedNodes():
        if n not in done:
            return n


def place_nodes(graph, type='median'):
    done = []
    previously_done = 0
    distance = graph.getDoubleProperty("distance")
    absolute = graph.getDoubleProperty("viewMetric")
    for n in graph.getNodes():
        absolute[n] = np.absolute(distance[n])
    while len(done) < len(graph.nodes()):
        for node in absolute.getSortedNodes():
            # node = closest(graph, done)
            if node not in done:
                graph, done = place_node(graph, done, node, type)
        print("actually done "+str(len(done)/len(graph.nodes()))+"%")
        if previously_done == len(done)/len(graph.nodes()): # Corrige un bug mais voir our corriger le bug
            for node in graph.nodes():
                if node not in done:
                    graph.delNode(node)
        else:
            previously_done = len(done) / len(graph.nodes())
    return graph