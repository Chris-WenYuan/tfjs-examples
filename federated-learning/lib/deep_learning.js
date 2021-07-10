const tf = require('@tensorflow/tfjs');
require('tfjs-node-save');
// require('@tensorflow/tfjs-node');

async function build_model(feature_num, class_num) {
    const model = tf.sequential(feature_num, class_num);
    model.add(tf.layers.dense({
        units: 16,
        inputShape: [feature_num],
        kernelInitializer: 'heNormal',
        activation: 'relu'
    }));
    model.add(tf.layers.dropout({rate: 0.25}));
    model.add(tf.layers.dense({
        units: 8,
        kernelInitializer: 'heNormal',
        activation: 'relu'
    }));
    model.add(tf.layers.dropout({rate: 0.25}));
    model.add(tf.layers.dense({
        units: class_num,
        activation: 'softmax'
    }));
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    //model.summary();

    return new Promise(resolve => {resolve(model)});
}

async function train_model(model, X_train, y_train, X_test, y_test) {
    const xs = tf.tensor2d(X_train, [X_train.length, X_train[0].length]);
    const ys = tf.tensor2d(y_train, [y_train.length, y_train[0].length]);
    const xsVal = tf.tensor2d(X_test, [X_test.length, X_test[0].length]);
    const ysVal = tf.tensor2d(y_test, [y_test.length, y_test[0].length]);

    // console.log(xs);
    // console.log(ys);

    const history = await model.fit(xs, ys, {
        epochs: 40,
        validationData: [xsVal, ysVal]
    });

    // const saveResults = await model.save('file://model/my-model-1');

    return new Promise(resolve => {resolve(model)});
}

function scale_model_weights(weight, scalar) {
    let weight_final = [];
    let steps = weight.length;
    console.log(steps);
    
    /*
    global_weight = global_model.getWeights();
    for (let i = 0; i < global_weight.length; i+=2) {
        console.log(global_weight[i].dataSync());
    }*/

    for(let i=0; i<steps; i++)
        weight_final.append(scalar * weight[i]);
    
    return weight_final;
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

    result_quantization = tf.zeros([result.length, result[0].length]).arraySync();

    for(var i=0; i<result.length; i++)
        result_quantization[i][indexOfMax(result[i])] = 1;

    return result_quantization;
}

// Compute the real anwser and prediction anwser, then calculate the accuracy.
function compare(y_test, y_predict) {
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
    build_model,
    train_model,
    scale_model_weights,
    load_model,
    predict,
    compare
};