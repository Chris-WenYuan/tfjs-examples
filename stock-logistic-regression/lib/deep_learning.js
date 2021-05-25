const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

async function train_model(X_train, X_test, y_train, y_test) {
    const xs = tf.tensor2d(X_train, [X_train.length, X_train[0].length]);
    const ys = tf.tensor2d(y_train, [y_train.length, y_train[0].length]);
    const xsVal = tf.tensor2d(X_test, [X_test.length, X_test[0].length]);
    const ysVal = tf.tensor2d(y_test, [y_test.length, y_test[0].length]);

    console.log(xs);
    console.log(ys);

    const model = tf.sequential();
    model.add(tf.layers.dense({units: 32, inputShape: [X_train[0].length], kernelInitializer: 'heNormal', activation: 'relu'}));
    model.add(tf.layers.dense({units: 8, kernelInitializer: 'heNormal', activation: 'relu'}));
    model.add(tf.layers.dense({units: y_train[0].length, kernelInitializer: 'heNormal'}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd', metrics: [tf.metrics.meanAbsoluteError]});
    model.summary();

    await model.fit(xs, ys, {
        epochs: 500,
        validationData: [xsVal, ysVal]
    });

    const saveResults = await model.save('file://model/my-model-1');

    return new Promise(resolve => {resolve(model)});
}

async function load_model(modelpath) {
    const model = await tf.loadLayersModel(modelpath);
    return new Promise(resolve => {resolve(model)});
}

// Find the index of maximum value in an array.
function indexOfMax(arr) {
    let max = arr[0];
    let maxIndex = 0;

    for(var i=0; i<arr.length; i++) {
        if(arr[i]>max) {
            max = arr[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

function predict(model, sample) {
    result = model.predict(tf.tensor2d(sample, [sample.length, sample[0].length])).arraySync();

    return result;
}

// Compute the real anwser and prediction anwser, then calculate the accuracy.
function evaluate_accuracy(y_test, y_predict) {
    correct = 0;
    for(var i=0; i<y_test.length; i++) {
        if(y_test[i].length == y_predict[i].length && 
            y_test[i].every(function(u, j) {
                return u === y_predict[i][j]
            })
        ) correct++;
    }
    console.log('Accuracy = ' + (correct / y_test.length)*100 + '%');
}

module.exports = {
    train_model,
    load_model,
    predict,
    evaluate_accuracy
};