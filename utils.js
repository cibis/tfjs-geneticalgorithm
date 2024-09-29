const tf = require('@tensorflow/tfjs-node');

function guidGenerator() {
    var S4 = function () {
        return (((1 + Math.random()) * 0x10000) | 0).toString(16).substring(1);
    };
    return (S4() + S4() + "-" + S4() + "-" + S4() + "-" + S4() + "-" + S4() + S4() + S4());
}

function optimizerBuilderFunction(optimizer, learningRate) {
    var res;
    switch (optimizer) {
        case "sgd":
            res = tf.train.sgd(learningRate)
            break;
        case "adagrad":
            res = tf.train.adagrad(learningRate)
            break;
        case "adadelta":
            res = tf.train.adadelta(learningRate)
            break;
        case "adam":
            res = tf.train.adam(learningRate)
            break;
        case "adamax":
            res = tf.train.adamax(learningRate)
            break;
        case "rmsprop":
            res = tf.train.rmsprop(learningRate)
            break;
    }
    return res;
}

module.exports = {
    guidGenerator,
    optimizerBuilderFunction
}