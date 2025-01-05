const tf = require('@tensorflow/tfjs-node');
const amqp = require("amqplib");
const http = require("http");

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
  constructor(host, path, port, options) {
      this.host = host;
      this.path = path;
      this.port = port;
      this.index = 0;
      this.options = options ?? {first : 0, last : 0};
      
      if(typeof this.options.first == 'undefined' || isNaN(this.options.first)) this.options.first = 0;
      this.options.first = parseFloat(this.options.first);
      
      if(typeof this.options.last == 'undefined' || isNaN(this.options.last)) this.options.last = 0; 
      this.options.last = parseFloat(this.options.last);        
  }

  getNextBatchFunction() {
      const iterator = {
          next: async () => {
              var options = {
                  host: this.host,
                  port: this.port,
                  path: `${this.path}?index=${this.index}&first=${this.options.first}&last=${this.options.last}`,
                  method: 'GET'
              };
        var asyncRequest = () => {
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
        var res = await asyncRequest(this.index);
        this.index++;

        return { value: { xs: tf.tensor(res.value.xs), ys: tf.tensor(res.value.ys) }, done: res.done };
      }
    };
    return iterator;
  }

}

async function trainModel(workerData) {
  try {
    var wguid = guidGenerator();
    console.log(`Worker ${wguid} started`);
    var phenotype = workerData.phenotype;
    // var tensorsAsArrays = JSON.parse(workerData.tensors);
    // var tensors = {
    //   trainFeatures: tf.tensor(tensorsAsArrays.trainFeatures),
    //   trainTarget: tf.tensor(tensorsAsArrays.trainTarget),
    //   testFeatures: tf.tensor(tensorsAsArrays.testFeatures),
    //   testTarget: tf.tensor(tensorsAsArrays.testTarget)
    // };
    const trainDataset =
      tf.data
        .generator(
          () => new DataSet(
            workerData.tensors.trainingDataSetSource.host,
            workerData.tensors.trainingDataSetSource.path,
            workerData.tensors.trainingDataSetSource.port, { first : (1 - workerData.validationSplit) * 100 }).getNextBatchFunction()
        );
    const trainValidationDataset =
      tf.data
        .generator(
          () => new DataSet(
            workerData.tensors.trainingDataSetSource.host,
            workerData.tensors.trainingDataSetSource.path,
            workerData.tensors.trainingDataSetSource.port, { last : workerData.validationSplit * 100 }).getNextBatchFunction()
        );        
    const valDataset =
      tf.data
        .generator(
          () => new DataSet(
            workerData.tensors.validationDataSetSource.host,
            workerData.tensors.validationDataSetSource.path,
            workerData.tensors.validationDataSetSource.port).getNextBatchFunction()
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

    console.log("Model training completed. loss " + testLoss);
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