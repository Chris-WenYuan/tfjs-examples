const fs = require('fs');
const csv = require('csv-parser');
const tf = require('@tensorflow/tfjs');

// Judge whether the string is a number or not.
function isNumeric(string) {
    if(typeof(string)!='string') return false;
    return !isNaN(string) && !isNaN(parseFloat(string));
}

// Read csv file according to the given filepath.
async function read_csv(filepath) {
    return new Promise(resolve => {
        let x = new Array();
        let y = new Array();

        let index = 0;
        fs.createReadStream(filepath)
            .pipe(csv())
            .on('data', (row) => {
                let values = Object.values(row);
                y.push(values.pop());
                x.push(values);
            })
            .on('end', () => {
                for(let i=0; i<x.length; i++) {
                    let values = x[i];
                    if(isNumeric(y[i])) y[i] = Number(y[i]);
                    for(let j=0; j<values.length; j++)
                        if(isNumeric(x[i][j])) x[i][j] = Number(x[i][j]);
                }
                resolve([x, y]);
            })
    });
}

// Normalize a 2D-array.
function normalize2d(array) {
    for(var j=0; j<array[0].length; j++) {
        let sum = 0;
        for(var i=0; i<array.length; i++) {
            sum += array[i][j];
        }
        let mean = sum / array.length;

        sum = 0;
        for(var i=0; i<array.length; i++) {
            sum += Math.pow(array[i][j] - mean, 2);
        }
        let std = Math.sqrt(sum / array.length);
        
        for(var i=0; i<array.length; i++) {
            array[i][j] = (array[i][j] - mean) / std;
        }
    }
    return array;
}

// One hot encode the given label array.
function one_hot(y_array) {
    const classes = y_array.filter((value, index, self) => {
        return self.indexOf(value) === index;
    });
    classes.sort();
    
    for(let i=0; i<y_array.length; i++)
        for(let j=0; j<classes.length; j++)
            if(y_array[i] == classes[j]) {
                y_array[i] = j;
                break;
            }

    y_array = tf.oneHot(tf.tensor1d(y_array, 'int32'), classes.length).arraySync();

    return y_array;
}

// Shuffle the array.
function shuffle(X_array, y_array) {
    var currentIndex = X_array.length, temporaryValue, randomIndex;

    // While there remain elements to shuffle...
    while(0!==currentIndex) {
        // Pick a remaining element...
        randomIndex = Math.floor(Math.random()*currentIndex);
        currentIndex--;

        // And swap it with the current element.
        temporaryValue = X_array[currentIndex];
        X_array[currentIndex] = X_array[randomIndex];
        X_array[randomIndex] = temporaryValue;
        temporaryValue = y_array[currentIndex];
        y_array[currentIndex] = y_array[randomIndex];
        y_array[randomIndex] = temporaryValue;
    }

    return X_array, y_array;
}

// Split dataset into training dataset and test dataset.
function train_test_split(X_array, y_array, train_size=1, random=false) {
    if(random) shuffle(X_array, y_array);

    const trainNum = Math.floor(X_array.length * train_size);
    const testNum = X_array.length - trainNum;

    let X_train = X_array.slice(0, trainNum);
    let X_test = X_array.slice(trainNum);
    let y_train = y_array.slice(0, trainNum);
    let y_test = y_array.slice(trainNum)

    return [X_train, X_test, y_train, y_test];
}

function create_clients(X_train, y_train, num_clients=10) {
    // Shard data and place at each client.
    let size = Math.floor(X_train.length / num_clients);
    let X_shards = [];
    let y_shards = [];

    for(let i=0; i<num_clients; i++) {
        X_shards.push([]);
        y_shards.push([]);
        for(let j=0; j<size; j++) {
            X_shards[i].push(X_train[i*size+j]);
            y_shards[i].push(y_train[i*size+j]);
        }
    }

    return [X_shards, y_shards];
}

module.exports = {
    read_csv,
    normalize2d,
    one_hot,
    train_test_split,
    create_clients
};