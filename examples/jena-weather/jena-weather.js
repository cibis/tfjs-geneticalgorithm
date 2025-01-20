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

/**
 * Data object for Jena Weather data.
 *
 * The data used in this demo is the
 * [Jena weather archive
 * dataset](https://www.kaggle.com/pankrzysiu/weather-archive-jena).
 * 
 * This file is used to load the Jena weather data in both
 * - the browser: see [index.js](./index.js), and 
 * - the Node.js backend environment: see [train-rnn.js](./train-rnn.js).
 */

const tf = require('@tensorflow/tfjs-node');

const LOCAL_JENA_WEATHER_CSV_PATH = './jena_climate_2009_2016.csv';
const REMOTE_JENA_WEATHER_CSV_PATH =
  'https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv';

/**
 * A class that fetches and processes the Jena weather archive data.
 *
 * It also provides a method to create a function that iterates over
 * batches of training or validation data.
 */
class JenaWeatherData {
  constructor() { }

  /**
   * Load and preprocess data.
   *
   * This method first tries to load the data from `LOCAL_JENA_WEATHER_CSV_PATH`
   * (a relative path) and, if that fails, will try to load it from a remote
   * URL (`JENA_WEATHER_CSV_PATH`).
   */
  async load() {
    let response;
    try {
      response = await fetch(LOCAL_JENA_WEATHER_CSV_PATH);
    } catch (err) { }

    if (response != null &&
      (response.statusCode === 200 || response.statusCode === 304)) {
      //console.log('Loading data from local path');
    } else {
      response = await fetch(REMOTE_JENA_WEATHER_CSV_PATH);
      // console.log(
      //   `Loading data from remote path: ${REMOTE_JENA_WEATHER_CSV_PATH}`);
    }
    const csvData = await response.text();

    // Parse CSV file.
    const csvLines = csvData.split('\n');

    // Parse header.
    const columnNames = csvLines[0].split(',');
    for (let i = 0; i < columnNames.length; ++i) {
      // Discard the quotes around the column name.
      columnNames[i] = columnNames[i].slice(1, columnNames[i].length - 1);
    }

    this.dateTimeCol = columnNames.indexOf('Date Time');
    tf.util.assert(this.dateTimeCol === 0, `Unexpected date-time column index`);

    this.dataColumnNames = columnNames.slice(1);
    this.tempCol = this.dataColumnNames.indexOf('T (degC)');
    tf.util.assert(this.tempCol >= 1, `Unexpected T (degC) column index`);

    this.dateTime = [];
    this.data = [];  // Unnormalized data.
    // Day of the year data, normalized between 0 and 1.
    this.normalizedDayOfYear = [];
    // Time of the day, normalized between 0 and 1.
    this.normalizedTimeOfDay = [];
    for (let i = 1; i < csvLines.length; ++i) {
      const line = csvLines[i].trim();
      if (line.length === 0) {
        continue;
      }
      const items = line.split(',');
      const parsed = this.parseDateTime_(items[0]);
      const newDateTime = parsed.date;
      if (this.dateTime.length > 0 &&
        newDateTime.getTime() <=
        this.dateTime[this.dateTime.length - 1].getTime()) {
      }

      this.dateTime.push(newDateTime);
      this.data.push(items.slice(1).map(x => +x));
      this.normalizedDayOfYear.push(parsed.normalizedDayOfYear);
      this.normalizedTimeOfDay.push(parsed.normalizedTimeOfDay);
    }
    this.numRows = this.data.length;
    this.numColumns = this.data[0].length;
    this.numColumnsExcludingTarget = this.data[0].length - 1;
    // console.log(
    //   `this.numColumnsExcludingTarget = ${this.numColumnsExcludingTarget}`);

    await this.calculateMeansAndStddevs_();
  }

  /**
   * Parse the date-time string from the Jena weather CSV file.
   *
   * @param {*} str The date time string with a format that looks like:
   *   "17.01.2009 22:10:00"
   * @returns date: A JavaScript Date object.
   *          normalizedDayOfYear: Day of the year, normalized between 0 and 1.
   *          normalizedTimeOfDay: Time of the day, normalized between 0 and 1.
   */
  parseDateTime_(str) {
    const items = str.split(' ');
    const dateStr = items[0];
    const dateStrItems = dateStr.split('.');
    const day = +dateStrItems[0];
    const month = +dateStrItems[1] - 1;  // month is 0-based in JS `Date` class.
    const year = +dateStrItems[2];

    const timeStrItems = items[1].split(':');
    const hours = +timeStrItems[0];
    const minutes = +timeStrItems[1];
    const seconds = +timeStrItems[2];

    const date = new Date(Date.UTC(year, month, day, hours, minutes, seconds));
    const yearOnset = new Date(year, 0, 1);
    const normalizedDayOfYear =
      (date - yearOnset) / (366 * 1000 * 60 * 60 * 24);
    const dayOnset = new Date(year, month, day);
    const normalizedTimeOfDay = (date - dayOnset) / (1000 * 60 * 60 * 24)
    return { date, normalizedDayOfYear, normalizedTimeOfDay };
  }


  /**
   * Calculate the means and standard deviations of every column.
   *
   * TensorFlow.js is used for acceleration.
   */
  async calculateMeansAndStddevs_() {
    tf.tidy(() => {
      // Instead of doing it on all columns at once, we do it
      // column by column, as doing it all at once causes WebGL OOM
      // on some machines.
      this.means = [];
      this.stddevs = [];
      for (const columnName of this.dataColumnNames) {
        // TODO(cais): See if we can relax this limit.
        const data =
          tf.tensor1d(this.getColumnData(columnName).slice(0, 6 * 24 * 365));
        const moments = tf.moments(data);
        this.means.push(moments.mean.dataSync());
        this.stddevs.push(Math.sqrt(moments.variance.dataSync()));
      }
      // console.log('means:', this.means);
      // console.log('stddevs:', this.stddevs);
    });

    // Cache normalized values.
    this.normalizedData = [];
    for (let i = 0; i < this.numRows; ++i) {
      const row = [];
      for (let j = 0; j < this.numColumns; ++j) {
        row.push((this.data[i][j] - this.means[j]) / this.stddevs[j]);
      }
      this.normalizedData.push(row);
    }
  }

  getDataColumnNames() {
    return this.dataColumnNames;
  }

  getTime(index) {
    return this.dateTime[index];
  }

  /** Get the mean and standard deviation of a data column. */
  getMeanAndStddev(dataColumnName) {
    if (this.means == null || this.stddevs == null) {
      throw new Error('means and stddevs have not been calculated yet.');
    }

    const index = this.getDataColumnNames().indexOf(dataColumnName);
    if (index === -1) {
      throw new Error(`Invalid data column name: ${dataColumnName}`);
    }
    return {
      mean: this.means[index], stddev: this.stddevs[index]
    }
  }

  getColumnData(
    columnName, includeTime, normalize, beginIndex, length, stride) {
    const columnIndex = this.dataColumnNames.indexOf(columnName);
    tf.util.assert(columnIndex >= 0, `Invalid column name: ${columnName}`);

    if (beginIndex == null) {
      beginIndex = 0;
    }
    if (length == null) {
      length = this.numRows - beginIndex;
    }
    if (stride == null) {
      stride = 1;
    }
    const out = [];
    for (let i = beginIndex; i < beginIndex + length && i < this.numRows;
      i += stride) {
      let value = normalize ? this.normalizedData[i][columnIndex] :
        this.data[i][columnIndex];
      if (includeTime) {
        value = { x: this.dateTime[i].getTime(), y: value };
      }
      out.push(value);
    }
    return out;
  }

  /**
   * Get a data iterator function.
   *
   * @param {boolean} shuffle Whether the data is to be shuffled. If `false`,
   *   the examples generated by repeated calling of the returned iterator
   *   function will scan through range specified by `minIndex` and `maxIndex`
   *   (or the entire range of the CSV file if those are not specified) in a
   *   sequential fashion. If `true`, the examples generated by the returned
   *   iterator function will start from random rows.
   * @param {number} lookBack Number of look-back time steps. This is how many
   *   steps to look back back when making a prediction. Typical value: 10 days
   *   (i.e., 6 * 24 * 10 = 1440).
   * @param {number} delay Number of time steps from the last time point in the
   *   input features to the time of prediction. Typical value: 1 day (i.e.,
   *   6 * 24 = 144).
   * @param {number} batchSize Batch size.
   * @param {number} step Number of steps between consecutive time points in the
   *   input features. This is a downsampling factor for the input features.
   *   Typical value: 1 hour (i.e., 6).
   * @param {number} minIndex Optional minimum index to draw from the original
   *   data set. Together with `maxIndex`, this can be used to reserve a chunk
   *   of the original data for validation or evaluation.
   * @param {number} maxIndex Optional maximum index to draw from the original
   *   data set. Together with `minIndex`, this can be used to reserve a chunk
   *   of the original data for validation or evaluation.
   * @param {boolean} normalize Whether the iterator function will return
   *   normalized data.
   * @param {boolean} includeDateTime Include the date-time features, including
   *   normalized day-of-the-year and normalized time-of-the-day.
   * @return {Function} An iterator Function, which returns a batch of features
   *   and targets when invoked. The features and targets are arranged in a
   *   length-2 array, in the said order.
   *   The features are represented as a float32-type `tf.Tensor` of shape
   *     `[batchSize, Math.floor(lookBack / step), featureLength]`
   *   The targets are represented as a float32-type `tf.Tensor` of shape
   *     `[batchSize, 1]`.
   */
  getNextBatchFunction(
    shuffle, lookBack, delay, batchSize, step, minIndex, maxIndex, normalize,
    includeDateTime) {
    let startIndex = minIndex + lookBack;
    const lookBackSlices = Math.floor(lookBack / step);

    return {
      next: () => {
        const rowIndices = [];
        let done = false;  // Indicates whether the dataset has ended.
        if (shuffle) {
          // If `shuffle` is `true`, start from randomly chosen rows.
          const range = maxIndex - (minIndex + lookBack);
          for (let i = 0; i < batchSize; ++i) {
            const row = minIndex + lookBack + Math.floor(Math.random() * range);
            rowIndices.push(row);
          }
        } else {
          // If `shuffle` is `false`, the starting row indices will be sequential.
          let r = startIndex;
          for (; r < startIndex + batchSize && r < maxIndex; ++r) {
            rowIndices.push(r);
          }
          if (r >= maxIndex) {
            done = true;
          }
        }

        const numExamples = rowIndices.length;
        //console.log(numExamples)
        startIndex += numExamples;

        const featureLength =
          includeDateTime ? this.numColumns + 2 : this.numColumns;
        const samples = tf.buffer([numExamples, lookBackSlices, featureLength]);
        const targets = tf.buffer([numExamples, 1]);
        // Iterate over examples. Each example contains a number of rows.
        for (let j = 0; j < numExamples; ++j) {
          const rowIndex = rowIndices[j];
          let exampleRow = 0;
          // Iterate over rows in the example.
          for (let r = rowIndex - lookBack; r < rowIndex; r += step) {
            let exampleCol = 0;
            // Iterate over features in the row.
            for (let n = 0; n < featureLength; ++n) {
              let value;
              if (n < this.numColumns) {
                value = normalize ? this.normalizedData[r][n] : this.data[r][n];
              } else if (n === this.numColumns) {
                // Normalized day-of-the-year feature.
                value = this.normalizedDayOfYear[r];
              } else {
                // Normalized time-of-the-day feature.
                value = this.normalizedTimeOfDay[r];
              }
              samples.set(value, j, exampleRow, exampleCol++);
            }

            const value = normalize ?
              this.normalizedData[r + delay][this.tempCol] :
              this.data[r + delay][this.tempCol];
            targets.set(value, j, 0);
            exampleRow++;
          }
        }
        return {
          value: { xs: samples.toTensor(), ys: targets.toTensor() },
          done
        };
      }
    };
  }
}


// Row ranges of the training and validation data subsets.
const TRAIN_MIN_ROW = 0;
const TRAIN_MAX_ROW = 200000;
const VAL_MIN_ROW = 200001;
const VAL_MAX_ROW = 300000;

/**
 * Calculate the commonsense baseline temperture-prediction accuracy.
 *
 * The latest value in the temperature feature column is used as the
 * prediction.
 *
 * @param {boolean} normalize Whether to used normalized data for training.
 * @param {boolean} includeDateTime Whether to include date and time features
 *   in training.
 * @param {number} lookBack Number of look-back time steps.
 * @param {number} step Step size used to generate the input features.
 * @param {number} delay How many steps in the future to make the prediction
 *   for.
 * @returns {number} The mean absolute error of the commonsense baseline
 *   prediction.
 */
async function getBaselineMeanAbsoluteError(
  jenaWeatherData, normalize, includeDateTime, lookBack, step, delay) {
  const batchSize = 128;
  const dataset = tf.data.generator(
    () => jenaWeatherData.getNextBatchFunction(
      false, lookBack, delay, batchSize, step, VAL_MIN_ROW, VAL_MAX_ROW,
      normalize, includeDateTime));

  const batchMeanAbsoluteErrors = [];
  const batchSizes = [];
  await dataset.forEachAsync(dataItem => {
    const features = dataItem.xs;
    const targets = dataItem.ys;
    const timeSteps = features.shape[1];
    batchSizes.push(features.shape[0]);
    batchMeanAbsoluteErrors.push(tf.tidy(
      () => tf.losses.absoluteDifference(
        targets,
        features.gather([timeSteps - 1], 1).gather([1], 2).squeeze([2]))));
  });

  const meanAbsoluteError = tf.tidy(() => {
    const batchSizesTensor = tf.tensor1d(batchSizes);
    const batchMeanAbsoluteErrorsTensor = tf.stack(batchMeanAbsoluteErrors);
    return batchMeanAbsoluteErrorsTensor.mul(batchSizesTensor)
      .sum()
      .div(batchSizesTensor.sum());
  });
  tf.dispose(batchMeanAbsoluteErrors);
  return meanAbsoluteError.dataSync()[0];
}

/**
 * Build a linear-regression model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @returns {tf.LayersModel} A TensorFlow.js tf.LayersModel instance.
 */
function buildLinearRegressionModel(inputShape) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape }));
  model.add(tf.layers.dense({ units: 1 }));
  return model;
}

/**
 * Build a GRU model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @param {tf.regularizer.Regularizer} kernelRegularizer An optional
 *   regularizer for the kernel of the first (hdiden) dense layer of the MLP.
 *   If not specified, no weight regularization will be included in the MLP.
 * @param {number} dropoutRate Dropout rate of an optional dropout layer
 *   inserted between the two dense layers of the MLP. Optional. If not
 *   specified, no dropout layers will be included in the MLP.
 * @returns {tf.LayersModel} A TensorFlow.js tf.LayersModel instance.
 */
function buildMLPModel(inputShape, kernelRegularizer, dropoutRate) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape }));
  model.add(
    tf.layers.dense({ units: 32, kernelRegularizer, activation: 'relu' }));
  if (dropoutRate > 0) {
    model.add(tf.layers.dropout({ rate: dropoutRate }));
  }
  model.add(tf.layers.dense({ units: 1 }));
  return model;
}

/**
 * Build a simpleRNN-based model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @returns {tf.LayersModel} A TensorFlow.js model consisting of a simpleRNN
 *   layer.
 */
function buildSimpleRNNModel(inputShape) {
  const model = tf.sequential();
  const rnnUnits = 32;
  model.add(tf.layers.simpleRNN({ units: rnnUnits, inputShape }));
  model.add(tf.layers.dense({ units: 1 }));
  return model;
}

/**
 * Build a GRU model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @param {number} dropout Optional input dropout rate
 * @param {number} recurrentDropout Optional recurrent dropout rate.
 * @returns {tf.LayersModel} A TensorFlow.js GRU model.
 */
function buildGRUModel(inputShape, dropout, recurrentDropout) {
  // TODO(cais): Recurrent dropout is currently not fully working.
  //   Make it work and add a flag to train-rnn.js.
  const model = tf.sequential();
  const rnnUnits = 32;
  model.add(tf.layers.gru({
    units: rnnUnits,
    inputShape,
    dropout: dropout || 0,
    recurrentDropout: recurrentDropout || 0
  }));
  model.add(tf.layers.dense({ units: 1 }));
  return model;
}

/**
 * Build a model for the temperature-prediction problem.
 *
 * @param {string} modelType Model type.
 * @param {number} numTimeSteps Number of time steps in each input.
 *   exapmle
 * @param {number} numFeatures Number of features (for each time step).
 * @returns A compiled instance of `tf.LayersModel`.
 */
function buildModel(modelType, numTimeSteps, numFeatures) {
  const inputShape = [numTimeSteps, numFeatures];

  console.log(`modelType = ${modelType}`);
  let model;
  if (modelType === 'mlp') {
    model = buildMLPModel(inputShape);
  } else if (modelType === 'mlp-l2') {
    model = buildMLPModel(inputShape, tf.regularizers.l2());
  } else if (modelType === 'linear-regression') {
    model = buildLinearRegressionModel(inputShape);
  } else if (modelType === 'mlp-dropout') {
    const regularizer = null;
    const dropoutRate = 0.25;
    model = buildMLPModel(inputShape, regularizer, dropoutRate);
  } else if (modelType === 'simpleRNN') {
    model = buildSimpleRNNModel(inputShape);
  } else if (modelType === 'gru') {
    model = buildGRUModel(inputShape);
    // TODO(cais): Add gru-dropout with recurrentDropout.
  } else {
    throw new Error(`Unsupported model type: ${modelType}`);
  }

  model.compile({ loss: 'meanAbsoluteError', optimizer: 'rmsprop' });
  model.summary();
  return model;
}

/**
 * Train a model on the Jena weather data.
 *
 * @param {tf.LayersModel} model A compiled tf.LayersModel object. It is
 *   expected to have a 3D input shape `[numExamples, timeSteps, numFeatures].`
 *   and an output shape `[numExamples, 1]` for predicting the temperature
 * value.
 * @param {JenaWeatherData} jenaWeatherData A JenaWeatherData object.
 * @param {boolean} normalize Whether to used normalized data for training.
 * @param {boolean} includeDateTime Whether to include date and time features
 *   in training.
 * @param {number} lookBack Number of look-back time steps.
 * @param {number} step Step size used to generate the input features.
 * @param {number} delay How many steps in the future to make the prediction
 *   for.
 * @param {number} batchSize batchSize for training.
 * @param {number} epochs Number of training epochs.
 * @param {tf.Callback | tf.CustomCallbackArgs} customCallback Optional callback
 *   to invoke at the end of every epoch. Can optionally have `onBatchEnd` and
 *   `onEpochEnd` fields.
 */
async function trainModel(
  model, jenaWeatherData, normalize, includeDateTime, lookBack, step, delay,
  batchSize, epochs, customCallback) {
  const trainShuffle = true;
  const trainDataset =
    tf.data
      .generator(
        () => jenaWeatherData.getNextBatchFunction(
          trainShuffle, lookBack, delay, batchSize, step, TRAIN_MIN_ROW,
          TRAIN_MAX_ROW, normalize, includeDateTime))
      .prefetch(8);

  const evalShuffle = false;
  const valDataset = tf.data.generator(
    () => jenaWeatherData.getNextBatchFunction(
      evalShuffle, lookBack, delay, batchSize, step, VAL_MIN_ROW,
      VAL_MAX_ROW, normalize, includeDateTime));

  await model.fitDataset(trainDataset, {
    epochs,
    callbacks: customCallback,
    validationData: valDataset
  });


  const result = await model.evaluateDataset(valDataset, {
    valDataset: 500
  });

  const testLoss = result.dataSync()[0].toFixed(4);
  console.log(`TEST LOSS: ${testLoss}`);
  // var tensors = {
  //     trainFeatures: [],
  //     trainTarget: [],
  // };
  // console.log("prepare data");
  // await trainDataset.forEachAsync(tc => { tensors.trainFeatures.push(tc.xs); tensors.trainTarget.push(tc.ys); });
  // console.log("model.fit");
  // await model.fit(tensors.trainFeatures, tensors.trainTarget, {
  //     epochs,
  //     callbacks: customCallback,
  //     validationSplit: 0.2
  // });
}


let jenaWeatherData;


async function run(modelType, jenaWeatherData) {
  const lookBack = 10 * 24 * 6;  // Look back 10 days.
  const step = 6;                // 1-hour steps.
  const delay = 24 * 6;          // Predict the weather 1 day later.
  const batchSize = 128;
  const normalize = true;
  const includeDateTime = false;
  const epochs = 20;

  console.log('Loading Jena weather data (41.2 MB)...');
  console.log('Done loading Jena weather data.');

  //var modelType = "mlp"; //"mlp-l2", "mlp-dropout", "linear-regression"
  let numFeatures = jenaWeatherData.getDataColumnNames().length;
  const model = buildModel(modelType, Math.floor(lookBack / step), numFeatures);

  console.log('Starting model training...');
  var trainLogs = [];

  await trainModel(
    model, jenaWeatherData, normalize, includeDateTime,
    lookBack, step, delay, batchSize, epochs, {
    onEpochEnd: async (epoch, logs) => {
      trainLogs.push(logs);
    }
  });

  console.log('Model training complete...');

  const trainLoss = trainLogs[trainLogs.length - 1].loss.toFixed(4);
  const valLoss = parseFloat(trainLogs[trainLogs.length - 1].val_loss.toFixed(4));
  console.log("\n============================================================");

  console.log(
    `Final train-set loss: ${trainLoss}\n` +
    `Final validation-set loss: ${valLoss}\n`);
  console.log("\n============================================================");

  return valLoss;
}

var runPredefinedModels = module.exports.runPredefinedModels = async function (jenaWeatherData) {  
  console.log("\nTraining/Testing models with predefined structure.");
  
  // console.log("\n==============================mlp==============================");
  // var bestLoss = await run("mlp", jenaWeatherData);

  // console.log("\n===========================mlp-l2===========================");
  // bestLoss = Math.min(bestLoss, await run("mlp-l2", jenaWeatherData));  

  // console.log("\n===========================mlp-dropout===========================");
  // bestLoss = Math.min(bestLoss, await run("mlp-dropout", jenaWeatherData)); 
  bestLoss = 1000
  console.log("\n===========================linear-regression===========================");
  bestLoss = Math.min(bestLoss, await run("linear-regression", jenaWeatherData)); 

  console.log("\n===========================simpleRNN===========================");
  bestLoss = Math.min(bestLoss, await run("simpleRNN", jenaWeatherData)); 

  console.log("\n===========================gru===========================");
  bestLoss = Math.min(bestLoss, await run("gru", jenaWeatherData)); 
  
  console.log("\n============================================================");
  return bestLoss;

}

module.exports.JenaWeatherData = JenaWeatherData;


// module.exports.loaded = (callback)=>{
//   if(callback) callback();
// };