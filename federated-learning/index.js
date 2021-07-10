const resolve = require('path').resolve;
const data_preprocess = require('./lib/data_preprocess');
const deep_learning = require('./lib/deep_learning');

const CSVPATH = './dataset/heart.csv';
const MODELPATH = 'file://model/my-model-1/model.json';

async function main() {
    // Read csv file and split into X_array(data) and y_array(label).
    // Please make sure the last column of csv file is the label.
    let [X_array, y_array] = await data_preprocess.read_csv(CSVPATH);

    // Normalize the data array.
    X_array = data_preprocess.normalize2d(X_array);

    // One hot encode the label array.
    y_array = data_preprocess.one_hot(y_array);
    
    // Split dataset into training dataset and test dataset according to the train_size parameter.
    let [X_train, X_test, y_train, y_test] = data_preprocess.train_test_split(X_array, y_array, train_size=0.8, random=true);

    // Create 10 clients' data shards.
    let [X_shards, y_shards] = data_preprocess.create_clients(X_train, y_train, num_clients=10);

    let global_model = await deep_learning.build_model(X_train[0].length, y_train[0].length);
    
    let comms_round = 10;
    for(let comm_round=0; comm_round<comms_round; comm_round++) {
        global_weights = global_model.getWeights();

        let scaled_local_weights_list = [];

        for(let i=0; i<10; i++) {
            let local_model = await deep_learning.build_model(X_train[0].length, y_train[0].length);
            local_model.setWeights(global_weights);

            local_model = await deep_learning.train_model(local_model,
                                                          X_shards[i],
                                                          y_shards[i],
                                                          X_test,
                                                          y_test)
            
            // Scale the model weights and add to list.
            let scalling_factor = X_shards[i].length / X_train.length;
            
        }
    }

    /*
    // Train model and use X_test, y_test as validation dataset.
    const model = await deep_learning.train_model(X_train, X_test, y_train, y_test);

    // Load trained model.
    // const model = await deep_learning.load_model(MODELPATH);

    // Predict classes according to the given model and X_test (test data).
    y_predict = deep_learning.predict(model, X_test);

    deep_learning.compare(y_test, y_predict);
    */
}

main();