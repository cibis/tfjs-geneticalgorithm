/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const fs = require('fs');
const Papa = require('papaparse');
const tf = require('@tensorflow/tfjs-node');

function determineMeanAndStddev(data) {
  const dataMean = data.mean(0);
  // TODO(bileschi): Simplify when and if tf.var / tf.std added to the API.
  const diffFromMean = data.sub(dataMean);
  const squaredDiffFromMean = diffFromMean.square();
  const variance = squaredDiffFromMean.mean(0);
  const dataStd = variance.sqrt();
  return { dataMean, dataStd };
}

function normalizeTensor(data, dataMean, dataStd) {
  return data.sub(dataMean).div(dataStd);
}

// Boston Housing data constants:
const BASE_URL =
  './csv/';

const TRAIN_FEATURES_FN = 'train-data.csv';
const TRAIN_TARGET_FN = 'train-target.csv';
const TEST_FEATURES_FN = 'test-data.csv';
const TEST_TARGET_FN = 'test-target.csv';

/**
 * Given CSV data returns an array of arrays of numbers.
 *
 * @param {Array<Object>} data Downloaded data.
 *
 * @returns {Promise.Array<number[]>} Resolves to data with values parsed as floats.
 */
const parseCsv = async (data) => {
  return new Promise(resolve => {
    data = data.map((row) => {
      return Object.keys(row).map(key => parseFloat(row[key]));
    });
    resolve(data);
  });
};

/**
 * Downloads and returns the csv.
 *
 * @param {string} filename Name of file to be loaded.
 *
 * @returns {Promise.Array<number[]>} Resolves to parsed csv data.
 */
const loadCsv = async (filename) => {
  return new Promise(resolve => {
    const url = `${BASE_URL}${filename}`;

    //console.log(`  * Downloading data from: ${url}`);
    const file = fs.createReadStream('./csv/' + filename);
    Papa.parse(file, {
      download: true,
      header: true,
      complete: (results) => {
        resolve(parseCsv(results['data']));
      }
    })
  });
};

/** Helper class to handle loading training and test data. */
class BostonHousingDataset {
  constructor() {
    // Arrays to hold the data.
    this.trainFeatures = null;
    this.trainTarget = null;
    this.testFeatures = null;
    this.testTarget = null;
  }

  get numFeatures() {
    // If numFeatures is accessed before the data is loaded, raise an error.
    if (this.trainFeatures == null) {
      throw new Error('\'loadData()\' must be called before numFeatures')
    }
    return this.trainFeatures[0].length;
  }

  /** Loads training and test data. */
  async loadData() {
    [this.trainFeatures, this.trainTarget, this.testFeatures, this.testTarget] =
      await Promise.all([
        loadCsv(TRAIN_FEATURES_FN), loadCsv(TRAIN_TARGET_FN),
        loadCsv(TEST_FEATURES_FN), loadCsv(TEST_TARGET_FN)
      ]);

    shuffle(this.trainFeatures, this.trainTarget);
    shuffle(this.testFeatures, this.testTarget);
  }
}

/**
 * Shuffles data and target (maintaining alignment) using Fisher-Yates
 * algorithm.flab
 */
function shuffle(data, target) {
  let counter = data.length;
  let temp = 0;
  let index = 0;
  while (counter > 0) {
    index = (Math.random() * counter) | 0;
    counter--;
    // data:
    temp = data[counter];
    data[counter] = data[index];
    data[index] = temp;
    // target:
    temp = target[counter];
    target[counter] = target[index];
    target[index] = temp;
  }
};

// Convert loaded data into tensors and creates normalized versions of the
// features.
function arraysToTensors() {
  tensors.rawTrainFeatures = tf.tensor2d(bostonData.trainFeatures);
  tensors.trainTarget = tf.tensor2d(bostonData.trainTarget);
  tensors.rawTestFeatures = tf.tensor2d(bostonData.testFeatures);
  tensors.testTarget = tf.tensor2d(bostonData.testTarget);
  // Normalize mean and standard deviation of data.
  let { dataMean, dataStd } =
    determineMeanAndStddev(tensors.rawTrainFeatures);

  tensors.trainFeatures = normalizeTensor(
    tensors.rawTrainFeatures, dataMean, dataStd);
  tensors.testFeatures =
    normalizeTensor(tensors.rawTestFeatures, dataMean, dataStd);
};

// Some hyperparameters for model training.
const NUM_EPOCHS = 200;
const BATCH_SIZE = 40;
const LEARNING_RATE = 0.01;

const bostonData = new BostonHousingDataset();
const tensors = {};

/**
 * Builds and returns Linear Regression Model.
 *
 * @returns {tf.Sequential} The linear regression model.
 */
function linearRegressionModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [bostonData.numFeatures], units: 1 }));

  //model.summary();
  return model;
};

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 1 hidden layer, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression model.
 */
function multiLayerPerceptronRegressionModel1Hidden() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [bostonData.numFeatures],
    units: 50,
    activation: 'sigmoid',
    kernelInitializer: 'leCunNormal'
  }));
  model.add(tf.layers.dense({ units: 1 }));

  //model.summary();
  return model;
};

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 2 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression model.
 */
function multiLayerPerceptronRegressionModel2Hidden() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [bostonData.numFeatures],
    units: 50,
    activation: 'sigmoid',
    kernelInitializer: 'leCunNormal'
  }));
  model.add(tf.layers.dense(
    { units: 50, activation: 'sigmoid', kernelInitializer: 'leCunNormal' }));
  model.add(tf.layers.dense({ units: 1 }));

  //model.summary();
  return model;
};


/**
 * Compiles `model` and trains it using the train data and runs model against
 * test data. Issues a callback to update the UI after each epoch.
 *
 * @param {tf.Sequential} model Model to be trained.
 *  weights.
 */
var run = module.exports.run = async function (model) {
  model.compile(
    { optimizer: tf.train.sgd(LEARNING_RATE), loss: 'meanSquaredError' });

  let trainLogs = [];

  await model.fit(tensors.trainFeatures, tensors.trainTarget, {
    verbose: false,
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        //console.log(`Epoch ${epoch + 1} of ${NUM_EPOCHS} completed.`);
        trainLogs.push(logs);
      }
    }
  });

  const result = model.evaluate(
    tensors.testFeatures, tensors.testTarget, { batchSize: BATCH_SIZE });
  const testLoss = result.dataSync()[0];

  const trainLoss = trainLogs[trainLogs.length - 1].loss;
  const valLoss = trainLogs[trainLogs.length - 1].val_loss;
  console.log(
    `Final train-set loss: ${trainLoss.toFixed(4)}\n` +
    `Final validation-set loss: ${valLoss.toFixed(4)}\n` +
    `Test-set loss: ${testLoss.toFixed(4)}`);

    return testLoss.toFixed(4);
};

function computeBaseline() {
  const avgPrice = tensors.trainTarget.mean();
  console.log(`Average price: ${avgPrice.dataSync()}`);
  const baseline = tensors.testTarget.sub(avgPrice).square().mean();
  console.log(`Baseline loss: ${baseline.dataSync()}`);
  const baselineMsg = `Baseline loss (meanSquaredError) is ${baseline.dataSync()[0].toFixed(2)}`;
  return baseline;
};

// var prepareTensors = module.exports.prepareTensors = async function() {
//   await bostonData.loadData();
//   arraysToTensors();
//   computeBaseline();
// }

var runPredefinedModels = module.exports.runPredefinedModels = async function() {
  console.log('\n');
  await bostonData.loadData();
  arraysToTensors();
  computeBaseline();
  
  console.log("\nTraining/Testing models with predefined structure.");
  
  console.log("\n==============================linearRegressionModel==============================");
  var model = linearRegressionModel();
  var bestLoss = await run(model);

  console.log("\n===========================multiLayerPerceptronRegressionModel1Hidden===========================");
  model = multiLayerPerceptronRegressionModel1Hidden();
  bestLoss = Math.min(bestLoss, await run(model));  

  console.log("\n===========================multiLayerPerceptronRegressionModel2Hidden===========================");
  model = multiLayerPerceptronRegressionModel2Hidden();
  bestLoss = Math.min(bestLoss, await run(model));  
  console.log("\n============================================================");
  return bestLoss;
}

module.exports.arraysToTensors = arraysToTensors;

module.exports.getBostonData = () => {
  return bostonData;
}

module.exports.getTensor = () => {
  return tensors;
}
