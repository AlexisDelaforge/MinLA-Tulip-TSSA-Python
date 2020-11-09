function abstract(data){
    var data_to_send = JSON.parse(JSON.stringify(data));
    // console.log(data_to_send)
    _.forEach(data_to_send.components, function(compo) {
        let value_nodes_0 = []
        let value_nodes_1 = []
        let value_nodes_f = []
        let min_distance = 0
        let max_distance = 0
        let slices = 20
        compo.slices = []
        _.forEach(compo.nodes, function (node) {
            // console.log(node)
            if(node.distance > max_distance){
                max_distance = node.distance
            }
            else if(node.distance < min_distance){
                min_distance = node.distance
            }
            if(node.classe == "0"){
                value_nodes_0.push(node.distance)
            }
            else if(node.classe == "1"){
                value_nodes_1.push(node.distance)
            }
            else{
                value_nodes_f.push(node.distance)
            }
            // console.log(value_nodes_0)
        })
        let amp = {min:min_distance, max:max_distance, amp:(max_distance-min_distance)}
        let steps = []
        steps.push(amp.min)
        let nb_steps_0 = []
        let nb_steps_1 = []
        for (let pas = 1; pas <= slices; pas++) {
            // Ceci sera exécuté 5 fois
            // À chaque éxécution, la variable "pas" augmentera de 1
            // Lorsque'elle sera arrivée à 5, le boucle se terminera.
            steps.push((amp.min+(amp.amp*pas)/slices))
            // console.log(value_nodes_0)
            nb_steps_0.push(
                value_nodes_0.filter(value => (value > steps.sort((a,b)=>a-b).reverse()[1] && value < steps.sort((a,b)=>a-b).reverse()[0])).length
            )
            nb_steps_1.push(
                value_nodes_1.filter(value => (value > steps.sort((a,b)=>a-b).reverse()[1] && value < steps.sort((a,b)=>a-b).reverse()[0])).length
            )
            // console.log(value_nodes_1.filter(value => (value > steps.reverse()[1] && value < steps.reverse()[0])).length)
        }
        // console.log(amp)
        // console.log(steps)

        delete compo.nodes;
        delete compo.edges;
        for (let pas = 1; pas < slices; pas++) {
            // console.log(compo.slices)
            compo.slices.push({
                mean:(steps[pas-1]+steps[pas])/2,
                min:steps[pas-1],
                max:steps[pas],
                classe0:nb_steps_0[pas-1],
                classe1:nb_steps_1[pas-1]
            })
        }
        // console.log(nb_steps_0)
        // console.log(nb_steps_1)

    })
    // console.log(data_to_send.components[0].slices)
    return data_to_send
}

function component(id){
    var component = data[id]
    return component
}

module.exports = {
    abstract: function(data) {
        return abstract(data);
    }
}