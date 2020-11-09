console.log('salut');

// Doc : https://nodejs.org/dist/latest-v12.x/docs/api/

let http = require('http');
let fs = require('fs');
let url = require('url');
const express = require('express');
const enableWs = require('express-ws')

const {spawn} = require('child_process');
const app = express();
const port = 3000;
enableWs(app)

// const python = spawn('python3.6', ['./python/MinLA.py', '../data/jeu_test.tlp']);


const python_script =
    {
        calcul1:"MinLA.py",
        calcul2:"median.py"
    }

app.get('/', (req, res) => {

    var dataToSend;
    // spawn new child process to call the python script
    console.log('start running code.')
    const python = spawn('python3.6', ['./python/MinLA.py', '../data/jeu_test.tlp']);
    // Then all you have to do is make sure that you import sys in your python script, and then you can access arg1 using sys.argv[1], arg2 using sys.argv[2], and so on.
    // collect data from script -> from the print of the python script
    // https://stackoverflow.com/questions/6758514/python-exit-codes
    python.stdout.on('data', function (data) {
        console.log('Pipe data from python MinLA script ...');
        dataToSend = data.toString();
        console.log(dataToSend)
    });
    // in close event we are sure that stream from child process is closed
    python.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
        // send data to browser
        // res.send(dataToSend)
        res.write('/median');
    });
    console.log('finish running code.')
    res.send('Calcul en cours.')
}).get('/median', (req, res) => {
    const python = spawn('python3.6', ['./python/median.py', '../data/jeu_test_done.tlp']);
    // Then all you have to do is make sure that you import sys in your python script, and then you can access arg1 using sys.argv[1], arg2 using sys.argv[2], and so on.
    // collect data from script -> from the print of the python script
    // https://stackoverflow.com/questions/6758514/python-exit-codes
    python.stdout.on('data', function (data) {
        console.log('Pipe data from python median script ...');
        dataToSend = data.toString();
        console.log(dataToSend)
    });
    // in close event we are sure that stream from child process is closed
    python.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
        // send data to browser
        // res.send(dataToSend)
        res.redirect('/finish');
    });
    console.log('finish running code.')
}).get('/finish', (req, res) => {
    // DO nothing
    res.render('./index.ejs', {data: fs.readFileSync('./data/my_test_perfect_json.json')})
    ws.on('message', msg => {
        ws.send(msg)
    })

    ws.on('close', () => {
        console.log('WebSocket was closed')
    })

})


app.listen(port, () => console.log(`Example app listening on port ${port}!`))

// let server = http.createServer(function (request, response) {
//     // "function (request, response)" === "(request, response) =>"
//     fs.readFile('index_old.html', (err, data) => {
//         if (err) {
//             response.writeHead(404);
//             response.end("Ce fichier n'existe pas.")
//         }
//         else {
//             // A voir pour chemin et données
//             // url_parsed = url.parse(request.url, true)
//             // console.log(url_parsed);
//             response.writeHead(200, {
//                 'Content-Type': 'text/html; charset=utf-8'
//             });
//             response.end(data);
//         }
//     })
// }).listen(port);

// http.get('http://nodejs.org/dist/index.json', (res) => {
//     const { statusCode } = res;
//     const contentType = res.headers['content-type'];
//
//     let error;
//     if (statusCode !== 200) {
//         error = new Error('Request Failed.\n' +
//             `Status Code: ${statusCode}`);
//     } else if (!/^application\/json/.test(contentType)) {
//         error = new Error('Invalid content-type.\n' +
//             `Expected application/json but received ${contentType}`);
//     }
//     if (error) {
//         console.error(error.message);
//         // Consume response data to free up memory
//         res.resume();
//         return;
//     }
//
//     res.setEncoding('utf8');
//     let rawData = '';
//     res.on('data', (chunk) => { rawData += chunk; });
//     res.on('end', () => {
//         try {
//             const parsedData = JSON.parse(rawData);
//             console.log(parsedData);
//         } catch (e) {
//             console.error(e.message);
//         }
//     });
// }).on('error', (e) => {
//     console.error(`Got error: ${e.message}`);
// });