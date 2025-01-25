const tf = require('@tensorflow/tfjs-node');
const amqp = require("amqplib");
const http = require("http");
const fs = require('fs');
var os = require("os");

const CACHE_STORAGE = "./_runtime/cache/"
const MODEL_STORAGE = "shared/models/"

const queueUrl = "amqp://rabbitmq-service:5672";
const inputQueue = process.env.JOB_NAME + "-INPUT";
const outputQueuePrefix = process.env.JOB_NAME + "-OUTPUT";

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

class DataSet {
  /**
   * 
   * @param {*} host 
   * @param {*} path 
   * @param {*} port 
   * @param {*} options { first: val, last: val }; first - Take only first portion of the data set. Acceptable value range 0 - 100; last - Take only last portion of the data set. Acceptable value range 0 - 100
   */
  constructor(host, path, port, cache_id, cache_batch_size, options) {
    this.host = host;
    this.path = path;
    this.port = port;
    this.cache_id = cache_id;
    this.cache_batch_size = cache_batch_size;
    this.index = 0;
    var defaultOptions = { first: 0, last: 0, batch_size: 32 };
    this.options = Object.assign(defaultOptions, options);
    if (parseInt(this.cache_batch_size) <= parseInt(this.options.batch_size))
      throw Error("cache_batch_size should always be bigger than options.batch_size");
  }

  async getNextBatchFunction() {
    var itemsCnt = 0;
    var lastLoadedCacheBatch = null;
    var lastLoadedCacheBatchIndex = null;
    if (this.cache_id) {
      var cacheFileDir = `${CACHE_STORAGE}${this.cache_id}/`;
      if (!fs.existsSync(cacheFileDir)) {
        var tmpFolderGuid = guidGenerator();
        cacheFileDir = `${CACHE_STORAGE}${tmpFolderGuid + "_"}${this.cache_id}/`;
        fs.mkdirSync(cacheFileDir, { recursive: true });
        var batch_index = 0;
        var asyncRequest = () => {
          var options = {
            host: this.host,
            port: this.port,
            path: `${this.path}?index=${batch_index}&cache_batch_size=${this.cache_batch_size}`,
            method: 'GET'
          };
          return new Promise(function (resolve, reject) {
            var req = http.get(options, function (res) {
              var bodyChunks = [];
              res.on('data', function (chunk) {
                bodyChunks.push(chunk);
              }).on('end', function () {
                var body = Buffer.concat(bodyChunks);
                resolve(JSON.parse(body));
              })
            });

            req.on('error', function (e) {
              console.log('ERROR: ' + e.message);
              reject(e);
            });
          });
        }
        var res;
        itemsCnt = 0;
        do {
          var res = await asyncRequest(batch_index);
          fs.writeFileSync(`${cacheFileDir}${batch_index}.json`, res.value);
          itemsCnt += JSON.parse(res.value).xs.length;
          batch_index++;
          if (res.done) {
            var firstBatch = JSON.parse(fs.readFileSync(`${cacheFileDir}0.json`));
            firstBatch.itemsCnt = itemsCnt;
            fs.writeFileSync(`${cacheFileDir}0.json`, JSON.stringify(firstBatch))
          }
        } while (!res.done);
        console.log('completed downloading. renaming folder');
        fs.renameSync(cacheFileDir, `${CACHE_STORAGE}${this.cache_id}/`);
        console.log('completed renaming folder');
        cacheFileDir = `${CACHE_STORAGE}${this.cache_id}/`;
      }

      this.options.first = parseFloat(this.options.first);
      this.options.last = parseFloat(this.options.last);

      if (!itemsCnt) {
        var entierCacheBatch = JSON.parse(fs.readFileSync(`${cacheFileDir}0.json`));
        itemsCnt = entierCacheBatch.itemsCnt;
      }
      var maxBatchIndex = Math.floor(itemsCnt / this.options.batch_size) - 1;

      var minBatchIndex = 0;

      if (this.options.first) {
        maxBatchIndex = Math.round(maxBatchIndex * this.options.first / 100);
      }
      if (this.options.last) {
        minBatchIndex = Math.round(maxBatchIndex * (100 - this.options.last) / 100) + 1;
      }

      this.index = minBatchIndex;
      const iterator = {

        next: async () => {
          var targetCacheBatchIndex = Math.floor(this.index * this.options.batch_size / this.cache_batch_size);
          var nextCacheBatchIndex = Math.floor((this.index + 1) * this.options.batch_size / this.cache_batch_size);
          var indexWithinTheCacheBatch = ((this.index * this.options.batch_size) % this.cache_batch_size);
          var moreBatchesInTheCacheBatch = indexWithinTheCacheBatch + this.options.batch_size < this.cache_batch_size;
          var entierCacheBatch = lastLoadedCacheBatch;
          if (lastLoadedCacheBatchIndex == null || targetCacheBatchIndex != lastLoadedCacheBatchIndex)
            entierCacheBatch = JSON.parse(fs.readFileSync(`${cacheFileDir}${targetCacheBatchIndex}.json`));
          lastLoadedCacheBatch = entierCacheBatch;
          lastLoadedCacheBatchIndex = targetCacheBatchIndex;
          var itemsLeftInTheCacheBatchToAdd = Math.min(entierCacheBatch.xs.length - indexWithinTheCacheBatch, this.options.batch_size);
          var batch = { xs: entierCacheBatch.xs.slice(indexWithinTheCacheBatch, indexWithinTheCacheBatch + itemsLeftInTheCacheBatchToAdd), ys: entierCacheBatch.ys.slice(indexWithinTheCacheBatch, indexWithinTheCacheBatch + itemsLeftInTheCacheBatchToAdd) };
          if (batch.xs.length < this.options.batch_size && !moreBatchesInTheCacheBatch && fs.existsSync(`${cacheFileDir}${nextCacheBatchIndex}.json`)) {
            //since cache batch is bigger than a training batch
            //next cache batch should have enough items for the training batch
            targetCacheBatchIndex++;
            entierCacheBatch = JSON.parse(fs.readFileSync(`${cacheFileDir}${targetCacheBatchIndex}.json`));
            lastLoadedCacheBatch = entierCacheBatch;
            lastLoadedCacheBatchIndex = targetCacheBatchIndex;
            batch.xs = batch.xs.concat(entierCacheBatch.xs.slice(0, this.options.batch_size - batch.xs.length));
            batch.ys = batch.ys.concat(entierCacheBatch.ys.slice(0, this.options.batch_size - batch.ys.length));
          }

          this.index++;
          var done = this.index > maxBatchIndex || (!moreBatchesInTheCacheBatch && !fs.existsSync(`${cacheFileDir}${nextCacheBatchIndex}.json`));
          return { value: { xs: tf.tensor(batch.xs), ys: tf.tensor(batch.ys) }, done: done };
        }
      };

      return iterator;
    }
  }

}

async function trainModel(workerData) {
  try {
    var wguid = guidGenerator();
    console.log(`Worker ${wguid} started`);
    var phenotype = workerData.phenotype;
    const trainDataset =
      tf.data
        .generator(
          () => new DataSet(
            workerData.tensors.trainingDataSetSource.host,
            workerData.tensors.trainingDataSetSource.path,
            workerData.tensors.trainingDataSetSource.port, 
            workerData.tensors.trainingDataSetSource.cache_id, 
            workerData.tensors.trainingDataSetSource.cache_batch_size, 
            { first : (1 - workerData.validationSplit) * 100, batch_size: phenotype.batchSize }
          ).getNextBatchFunction()
        );
    const trainValidationDataset =
      tf.data
        .generator(
          () => new DataSet(
            workerData.tensors.trainingDataSetSource.host,
            workerData.tensors.trainingDataSetSource.path,
            workerData.tensors.trainingDataSetSource.port, 
            workerData.tensors.trainingDataSetSource.cache_id, 
            workerData.tensors.trainingDataSetSource.cache_batch_size, 
            { last : workerData.validationSplit * 100, batch_size: phenotype.batchSize }
          ).getNextBatchFunction()
        );        
    const valDataset =
      tf.data
        .generator(
          () => new DataSet(
            workerData.tensors.validationDataSetSource.host,
            workerData.tensors.validationDataSetSource.path,
            workerData.tensors.validationDataSetSource.port,
            workerData.tensors.validationDataSetSource.cache_id, 
            workerData.tensors.validationDataSetSource.cache_batch_size, 
            { batch_size: phenotype.batchSize }
          ).getNextBatchFunction()
        );

    var modelAbortThreshold = workerData.modelAbortThreshold;
    var modelTrainingTimeThreshold = workerData.modelTrainingTimeThreshold;
    var validationSplit = workerData.validationSplit;

    var lossThresholdAbortCnt = 0;
    var model = null;
    do {
      const modelData = JSON.parse(workerData.modelJson);
      const weightData = new Uint8Array(Buffer.from(modelData.weightData, "base64")).buffer;
      model = await tf.loadLayersModel(tf.io.fromMemory({
        modelTopology: modelData.modelTopology,
        weightSpecs: modelData.weightSpecs,
        weightData: weightData
      }));

      var trainLogs = [];
      var lossThresholdAbort = false;
      var errorAbort = false;
      var epochTimeStart;

      model.compile(
        { optimizer: optimizerBuilderFunction(phenotype.optimizer, phenotype.learningRate), loss: phenotype.loss });
      var trainingStartTime = Date.now();
      await model.fitDataset(
        trainDataset,
        //tensors.trainFeatures, tensors.trainTarget,
        {
          verbose: false,
          //batchSize: phenotype.batchSize,
          epochs: phenotype.epochs,
          //validationSplit: validationSplit,
          validationData: trainValidationDataset,
          callbacks: {
            onEpochBegin: (epoch, logs) => {
              epochTimeStart = Date.now();
            },
            onEpochEnd: async (epoch, logs) => {
              trainLogs.push(logs);
              if (isNaN(logs.val_loss)) {
                console.log(`Early model loss is NaN abort. Epoch ${epoch} `);
                errorAbort = true;
                throw Error("Loss is NaN");
              }
              if (modelTrainingTimeThreshold && Date.now() > trainingStartTime + modelTrainingTimeThreshold * 1000) {
                console.log(`Early model training timeout abort. Epoch ${epoch} `);
                errorAbort = true;
                throw Error("Model training timeout abort");
              }
              if (modelTrainingTimeThreshold && epochTimeStart && (phenotype.epochs - 1 > epoch)
                && ((Date.now() - epochTimeStart) * (phenotype.epochs - epoch - 1)) > modelTrainingTimeThreshold * 1000) {
                console.log(`Early model training timeout abort based on prior epoch time. Epoch ${epoch} `);
                errorAbort = true;
                throw Error("Model training timeout abort");
              }  
              if (modelAbortThreshold && trainLogs.length > modelAbortThreshold && trainLogs[trainLogs.length - modelAbortThreshold].val_loss <= logs.val_loss) {
                //console.log(`Early model training abort(${lossThresholdAbortCnt+1}). Epoch ${epoch}. loss compare ` + trainLogs[trainLogs.length - modelAbortThreshold].val_loss + " <= " + logs.val_loss);
                lossThresholdAbort = true;
                throw Error("Early training abort");
              }
            }
          }
        }).catch((err) => {
          if (!lossThresholdAbort && !errorAbort) console.log(err)
        });

      if (errorAbort) {
        return { validationLoss: NaN, phenotype: phenotype };
      }
      if (lossThresholdAbort) {
        phenotype.epoch = trainLogs.length - modelAbortThreshold + 1;

        lossThresholdAbortCnt++;
        if (lossThresholdAbortCnt > 1) {
          console.log(`Early model training abort`);
          return { validationLoss: NaN, phenotype: phenotype };
        }
      }
    } while (lossThresholdAbort)


    const result = await model.evaluateDataset(
      valDataset
      //tensors.testFeatures, tensors.testTarget, { batchSize: phenotype.batchSize }
    );
    const testLoss = parseFloat(result.dataSync()[0].toFixed(4));
    const trainLoss = trainLogs[trainLogs.length - 1].loss.toFixed(4);
    const valLoss = trainLogs[trainLogs.length - 1].val_loss.toFixed(4);

    phenotype.validationLoss = testLoss;

    console.log(`Model training completed ${phenotype._id} . post training eval loss ${testLoss}, training validation-set loss: ${valLoss}, train-set loss: ${trainLoss}`);
    console.log("============================================================");

    let resultingModelData = await model.save(tf.io.withSaveHandler(async modelArtifacts => modelArtifacts));
    resultingModelData.weightData = Buffer.from(resultingModelData.weightData).toString("base64");
    const jsonStr = JSON.stringify(resultingModelData);

    return { validationLoss: testLoss, phenotype: phenotype, modelJson: jsonStr };
  }
  catch (err) {
    console.log(`Error: ${err} stack: ${err.stack}`)
    return { validationLoss: NaN, phenotype: phenotype };
  }
}

async function readQueue() {
  console.log(`${Date.now()} readQueue() start`);
  return new Promise(async function (resolve, reject) {
    try {
      const connection = await amqp.connect(queueUrl);
      const channel = await connection.createChannel();

      process.once("SIGINT", async () => {
        await channel.close();
        await connection.close();
        resolve(null);
      });

      await channel.assertQueue(inputQueue, { durable: false });
      await channel.prefetch(1);

      await channel.consume(
        inputQueue,
        async (message) => {
          if (message) {
            resolve(JSON.parse(message.content.toString()));
          }
          else {
            resolve(null);
          }
          await channel.ack(message);
          await channel.close();
          await connection.close();
          console.log(`${Date.now()} readQueue() closing channel`);
        }
      );

    } catch (err) {
      reject(err);
    }
  });
}


async function writeQueue(id, tfjsJobResponse) {
  return new Promise(async function (resolve, reject) {
    var outputQueue = `${outputQueuePrefix}-${id}`;
    console.log(`${Date.now()} writeQueue ${outputQueue}`);
    let connection;
    try {
      connection = await amqp.connect(queueUrl);
      const channel = await connection.createChannel();

      await channel.assertQueue(outputQueue, { durable: false });
      channel.sendToQueue(outputQueue, Buffer.from(JSON.stringify(tfjsJobResponse)));
      await channel.close();
      resolve();
    } catch (err) {
      console.warn(err);
      reject(err);
    } finally {
      if (connection) await connection.close();
    }
  });
}

async function main() {
  console.log(`${Date.now()} main() start`);
  while (true) {
    try {
      var tfjsJob = await readQueue();
      var response = await trainModel(tfjsJob.workerData);
      await writeQueue(response.phenotype._id, response);
    }
    catch (err) {
      console.log(`Worker trainModel Error: ${err} stack: ${err.stack}`)
    }
  }
}

main();