const resolve = require('path').resolve;
const data_preprocess = require('./lib/data_preprocess');
const deep_learning = require('./lib/deep_learning');

const CSVPATH = './dataset/TSLA.csv';
const MODELPATH = 'file://model/my-model-1/model.json';

async function main() {
    // Read a csv file and return as an array.
    let dataset = await data_preprocess.read_csv(CSVPATH);

    // Given the indexOf_X and indexOf_y, split the input array(dataset) into X and y.
    let [X, y] = data_preprocess.Xy_split(dataset, indexOf_X=[1, 2, 3, 6], indexOf_y=[4]);

    // Normalize the data array.
    X = data_preprocess.normalize2d(X);
    y = data_preprocess.normalize2d(y);

    // Split dataset into training dataset and test dataset according to the train_size parameter.
    let [X_train, X_test, y_train, y_test] = data_preprocess.train_test_split(X, y, train_size=0.8, random=true);

    // Train model and use X_test, y_test as validation dataset.
    const model = await deep_learning.train_model(X_train, X_test, y_train, y_test);

    // Load trained model.
    // const model = await deep_learning.load_model(MODELPATH);

    // Predict classes according to the given model and X_test (test data).
    y_predict = deep_learning.predict(model, X_test);

    console.log(y_test);
    console.log(y_predict);
}

main();