var socket = io();

// On demande le pseudo au visiteur...
// var pseudo = prompt('Quel est votre pseudo ?');
// Et on l'envoie avec le signal "petit_nouveau" (pour le différencier de "message")
// socket.emit('petit_nouveau', pseudo);

// On affiche une boîte de dialogue quand le serveur nous envoie un "message"
socket.on('message', function(message) {
    alert('Le serveur a un message pour vous : ' + message);
})

// Demande les données pour commencer
socket.emit('ask_abstracted_data', null);

function ask_compo_data(id){
    socket.emit('ask_compo_data', id);
    alert('Les données de la compo ' + id +'ont étaients demandés.');
    var newDiv = document.createElement("div");
    var body = document.body
    body.appendChild(newDiv);
    newDiv.id = "compo"+id
    create_modal(newDiv, id)

}

socket.on('reply_compo_data', function(compo_data) {
    alert('Les données de la compo : ' + compo_data);
    compo_data = JSON.parse(compo_data)
    compos_data[compo_data.name] = compo_data
    console.log(compos_data)
    // document.getElementById('#compo'+compo_data.name).modal('show')
    $('#compo'+compo_data.name).modal('show')
})


// Réception des données abstraites pour commencer
socket.on('reply_abstracted_data', function(message) {
    var data = JSON.parse(message);
    console.log(data)
    nb_compo = 0;
    _.forEach(data.components, function(values_compo, id_compo) {
        svg.append("g")
            .attr("id", 'grp' + id_compo)
            .attr("transform", "translate(" + ((nb_compo % nb_per_line) * grpWidth + grpMarginLeft) + "," + (Math.floor(nb_compo / nb_per_line) * grpHeight + grpMarginTop) +")")
        // .attr("cx", ((nb_compo % nb_per_line) * grpWidth))
        // .attr("cy", Math.floor(nb_compo / nb_per_line) * h);
        create_histograme(values_compo, id_compo)
        nb_compo++;
    });
})