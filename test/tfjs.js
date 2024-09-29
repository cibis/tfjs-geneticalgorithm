var assert = require('assert');
const tf = require('@tensorflow/tfjs-node');
describe('Tensorflow', function () {
    describe('tfjs-node is properly configured', function () {
        it('', function (done) {
            const model = tf.sequential();
            model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [10] }));
            model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
            model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

            const xs = tf.randomNormal([100, 10]);
            const ys = tf.randomNormal([100, 1]);

            // Train the model.
            model.fit(xs, ys, {
                epochs: 100,
                verbose: false,
                callbacks: {
                    onTrainEnd: async (logs) => {
                        done();
                    }
                }
            });
        });
    });    
});