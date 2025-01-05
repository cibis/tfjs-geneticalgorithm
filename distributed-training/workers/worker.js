const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const tf = require('@tensorflow/tfjs-node');
var ModelStorage = require("../../model-storage/current")
var utils = require("../../utils")
var DataService = require('../../data-service');

function guidGenerator() {
  var S4 = function () {
      return (((1 + Math.random()) * 0x10000) | 0).toString(16).substring(1);
  };
  return (S4() + S4() + "-" + S4() + "-" + S4() + "-" + S4() + "-" + S4() + S4() + S4());
}

async function trainModel() {
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
          () => new DataService.DataSet(
            workerData.tensors.trainingDataSetSource.host,
            workerData.tensors.trainingDataSetSource.path,
            workerData.tensors.trainingDataSetSource.port, { first : (1 - workerData.validationSplit) * 100 }).getNextBatchFunction()
        );
    const trainValidationDataset =
      tf.data
        .generator(
          () => new DataService.DataSet(
            workerData.tensors.trainingDataSetSource.host,
            workerData.tensors.trainingDataSetSource.path,
            workerData.tensors.trainingDataSetSource.port, { last : workerData.validationSplit * 100 }).getNextBatchFunction()
        );        
    const valDataset =
      tf.data
        .generator(
          () => new DataService.DataSet(
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
      model = await ModelStorage.readModel(phenotype._id);
      var trainLogs = [];
      var lossThresholdAbort = false;
      var errorAbort = false;

      model.compile(
        { optimizer: utils.optimizerBuilderFunction(phenotype.optimizer, phenotype.learningRate), loss: phenotype.loss });
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
        if(lossThresholdAbortCnt > 1){
          console.log(`Early model training abort`);
          return { validationLoss: NaN, phenotype: phenotype };
        }

        //return { validationLoss: trainLogs[trainLogs.length - modelAbortThreshold].val_loss, phenotype: phenotype };
      }
    } while (lossThresholdAbort)

    
    const result = await model.evaluateDataset(
      valDataset
      //tensors.testFeatures, tensors.testTarget, { batchSize: phenotype.batchSize }
    );
    const testLoss = parseFloat(result.dataSync()[0].toFixed(4));
    const trainLoss = trainLogs[trainLogs.length - 1].loss.toFixed(4);
    const valLoss = trainLogs[trainLogs.length - 1].val_loss.toFixed(4);
    // console.log("\n============================================================");
    // model.summary();
    // console.log(phenotype);
    //   console.log(
    //     `Final train-set loss: ${trainLoss}\n` +
    //     `Final validation-set loss: ${valLoss}\n` +
    //     `Test-set loss: ${testLoss}`);
    // console.log("\n============================================================");

    // var savedModel = await model.save(tf.io.withSaveHandler(async modelArtifacts => modelArtifacts));
    // savedModel.weightData = Buffer.from(savedModel.weightData).toString("base64");
    // const modelJson = JSON.stringify(savedModel);
    phenotype.validationLoss = testLoss;
    await ModelStorage.writeModel(phenotype._id, model);
    
    console.log(`Model training completed. post training eval loss ${testLoss}, training validation-set loss: ${valLoss}, train-set loss: ${trainLoss}`);

    return { validationLoss: testLoss, phenotype: phenotype/*,  modelJson: modelJson*/ };
  }
  catch (err) {
    console.log(`Error: ${err} stack: ${err.stack}`)
    return { validationLoss: NaN, phenotype: phenotype };
  }
}

trainModel().then(response => {
  //console.log(JSON.stringify(response))
  parentPort.postMessage(response);
}).catch(err=>{ console.log(`Worker trainModel Error: ${err} stack: ${err.stack}`) })