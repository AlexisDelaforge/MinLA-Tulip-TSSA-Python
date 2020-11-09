var http = require('http');
var fs = require('fs');
var port = 8000

// Chargement du fichier index.html affiché au client
var server = http.createServer(function(req, res) {

    fs.readFile('./index.html', 'utf-8', function(error, content) {
        res.writeHead(200, {"Content-Type": "text/html"});
        res.send
        res.end(content);
    });
});

// Chargement de socket.io
var io = require('socket.io').listen(server);

// Chargement de lodash
var _ = require('lodash');

// Chargement des données
const data = JSON.parse(fs.readFileSync('./data/my_test_perfect_json.json'))
// console.log(data)

// Modificationd des données

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

io.sockets.on('connection', function (socket, pseudo) {

    // Envoi des données abstraites
    socket.emit('reply_abstracted_data', JSON.stringify(abstract(data)))

    // Quand un client se connecte, on lui envoie un message
    socket.emit('message', 'Vous êtes bien connecté !');
    // On signale aux autres clients qu'il y a un nouveau venu
    socket.broadcast.emit('message', 'Un autre client vient de se connecter ! ');

    // Dès qu'on nous donne un pseudo, on le stocke en variable de session
    socket.on('petit_nouveau', function(pseudo) {
        socket.pseudo = pseudo;
    });

    socket.on('ask_compo_data', function(id) {
        socket.id = id;
        console.log('compo '+socket.id + ' est demandée');
        socket.emit('reply_compo_data', JSON.stringify(data.components[id]));
    });

    // Dès qu'on reçoit un "message" (clic sur le bouton), on le note dans la console
    socket.on('message', function (message) {
        // On récupère le pseudo de celui qui a cliqué dans les variables de session
        console.log(socket.pseudo + ' me parle ! Il me dit : ' + message);
    });
});
server.listen(port);
console.log('http://localhost:'+port);