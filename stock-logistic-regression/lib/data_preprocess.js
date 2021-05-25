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
    let dataset = new Array();
    return new Promise(resolve => {
        fs.createReadStream(filepath)
            .pipe(csv())
            .on('data', (row) => {
                dataset.push(Object.values(row));
            })
            .on('end', () => {
                for(var i=0; i<dataset.length; i++)
                    for(var j=0; j<dataset[0].length; j++)
                        if(isNumeric(dataset[i][j])) dataset[i][j] = Number(dataset[i][j]);
                resolve(dataset);
            })
    });
}

function Xy_split(array, indexOf_X, indexOf_y) {
    let X = new Array();
    let y = new Array();

    for(var i=0; i<array.length; i++) {
        X[i] = new Array();
        y[i] = new Array();
        for(var j=0; j<indexOf_X.length; j++)
            X[i].push(array[i][indexOf_X[j]]);
        for(var j=0; j<indexOf_y.length; j++)
            y[i].push(array[i][indexOf_y[j]]);
    }
    
    return [X, y];
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
function shuffle(x_array, y_array) {
    var currentIndex = x_array.length, temporaryValue, randomIndex;

    // While there remain elements to shuffle...
    while(0!==currentIndex) {
        // Pick a remaining element...
        randomIndex = Math.floor(Math.random()*currentIndex);
        currentIndex--;

        // And swap it with the current element.
        temporaryValue = x_array[currentIndex];
        x_array[currentIndex] = x_array[randomIndex];
        x_array[randomIndex] = temporaryValue;
        temporaryValue = y_array[currentIndex];
        y_array[currentIndex] = y_array[randomIndex];
        y_array[randomIndex] = temporaryValue;
    }

    return x_array, y_array;
}

// Split dataset into training dataset and test dataset.
function train_test_split(x_array, y_array, train_size=1, random=false) {
    if(random) shuffle(x_array, y_array);

    const trainNum = Math.floor(x_array.length * train_size);
    const testNum = x_array.length - trainNum;

    let X_train = x_array.slice(0, trainNum);
    let X_test = x_array.slice(trainNum);
    let y_train = y_array.slice(0, trainNum);
    let y_test = y_array.slice(trainNum)

    return [X_train, X_test, y_train, y_test];
}

module.exports = {
    read_csv,
    Xy_split,
    normalize2d,
    one_hot,
    train_test_split
};