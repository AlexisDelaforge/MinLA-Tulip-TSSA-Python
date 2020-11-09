from tulip import tlp
import numpy
import json
from scipy import stats


graph = tlp.loadGraph('test_V9_done_mean.tlp', None)
viewLayout = graph.getLayoutProperty("viewLayout")
classe = graph.getStringProperty("Classe")
predict =graph.getBooleanProperty("Predict")

print(graph)

components = tlp.ConnectedTest.computeConnectedComponents(graph)

data = {}
data['date'] = "28 janvier"
data['nb_components'] = len(components)
data['components'] = {}
for i in range(len(components)):
    data['components'][i] = {}
    data['components'][i]['name'] = str(i)
    data['components'][i]['nb_nodes'] = 0
    data['components'][i]['nb_edges'] = 0
    data['components'][i]['nb_nodes_frontier'] = 0
    data['components'][i]['nb_good_predcition'] = 0
    data['components'][i]['classes'] = [0, 1]
    data['components'][i]['nb_classes'] = 2
    data['components'][i]['nb_by_classes'] = dict()
    for clas in data['components'][i]['classes']:
        data['components'][i]['nb_by_classes'][str(clas)] = 0
    data['components'][i]['mean'] = 0
    data['components'][i]['name'] = str(i)
    # print("component "+str(components[i]))
    data['components'][i]['nodes'] = []
    data['components'][i]['edges'] = []
    values = []
    for node in components[i]:
        values.append(viewLayout[node][0])
        data['components'][i]['mean'] += viewLayout[node][0]
        y_min = float("inf")
        data['components'][i]['nb_nodes'] += 1
        if str(classe[node]) == str(data['components'][i]['classes'][0]):
            data['components'][i]['nb_by_classes'][str(data['components'][i]['classes'][0])] += 1
        elif str(classe[node]) == str(data['components'][i]['classes'][1]):
            data['components'][i]['nb_by_classes'][str(data['components'][i]['classes'][1])] += 1
        elif str(classe[node]) == 'frontier':
            data['components'][i]['nb_nodes_frontier'] += 1
        if str(predict[node]):
            data['components'][i]['nb_good_predcition'] += 1
        data['components'][i]['nodes'].append({
            'name': str(node.id),
            'classe': classe[node],
            'distance': viewLayout[node][0],
            'pos_y': viewLayout[node][1]
        })
        # data['components'][i]['nodes'][str(node.id)]['classe'] = classe[node]
        # data['components'][i]['nodes'][str(node.id)]['distance'] = viewLayout[node][0]
        # data['components'][i]['nodes'][str(node.id)]['pos_y'] = viewLayout[node][1]
        if viewLayout[node][1] < y_min:
            y_min = viewLayout[node][1]
        for neighbour in graph.getInOutNodes(node):
            if int(str(node.id)) < int(str(neighbour.id)):
                data['components'][i]['edges'].append([str(node.id), str(neighbour.id)])
                data['components'][i]['nb_edges'] += 1
    for node in data['components'][i]['nodes']:
        node['pos_y'] -= y_min # A VOIR POUR SUPPRIMER

    data['components'][i]['mean'] = data['components'][i]['mean']/data['components'][i]['nb_nodes']
    data['components'][i]['shapiro-wilk'] = stats.shapiro(values)[0]

print(data)
json_data = json.dumps(data)
fh = open("./my_test_perfect_json.json", "w+")
fh.write(json_data)