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

function initConsoleLogTimestamps(){
    var log = console.log;
    
    console.log = function () {
        var first_parameter = arguments[0];
        var other_parameters = Array.prototype.slice.call(arguments, 1);
    
        function formatConsoleDate (date) {
            var hour = date.getHours();
            var minutes = date.getMinutes();
            var seconds = date.getSeconds();
            var milliseconds = date.getMilliseconds();
    
            return '[' +
                   ((hour < 10) ? '0' + hour: hour) +
                   ':' +
                   ((minutes < 10) ? '0' + minutes: minutes) +
                   ':' +
                   ((seconds < 10) ? '0' + seconds: seconds) +
                   '.' +
                   ('00' + milliseconds).slice(-3) +
                   '] ';
        }
    
        log.apply(console, [formatConsoleDate(new Date()) + first_parameter].concat(other_parameters));
    };
}

module.exports = {
    guidGenerator,
    optimizerBuilderFunction,
    initConsoleLogTimestamps
}