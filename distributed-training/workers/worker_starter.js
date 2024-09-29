const { Worker, isMainThread, parentPort } = require('worker_threads');

var DistributedTrainingInterface = require("../DistributedTrainingInterface");

module.exports = class WorkerTraining extends DistributedTrainingInterface {
    constructor() {
        super();

    }

    trainModel(phenotype/*, modelJson*/, tensors, validationSplit, modelAbortThreshold, modelTrainingTimeThreshold) {
        return new Promise((resolve, reject) => {
            var runTask = async function (resolve, reject) {
                try {

                    const worker = new Worker("./distributed-training/workers/worker.js", {
                        workerData: { 
                            phenotype: phenotype, 
                            //modelJson : modelJson, 
                            tensors: JSON.stringify({ 
                                trainFeatures: tensors.trainFeatures.arraySync(), 
                                trainTarget: tensors.trainTarget.arraySync(), 
                                testFeatures: tensors.testFeatures.arraySync(), 
                                testTarget: tensors.testTarget.arraySync()
                            }), 
                            validationSplit: validationSplit, 
                            modelAbortThreshold: modelAbortThreshold,
                            modelTrainingTimeThreshold: modelTrainingTimeThreshold
                        },
                      });
                      worker.on("message", (data) => {
                        //console.log(data);
                        resolve(data);
                        worker.terminate();
                      });
                      worker.on("error", (msg) => {
                        reject(`An error ocurred: ${msg}`);
                        worker.terminate();
                      });
                }
                catch (e) { reject(e) }                    
            }            
            runTask(resolve, reject);
        })
    }
}