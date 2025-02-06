const amqp = require("amqplib");
const tf = require('@tensorflow/tfjs-node');
var ModelStorage = require("../../model-storage/current")

const queueUrl = "amqp://127.0.0.1:30000";


var DistributedTrainingInterface = require("../DistributedTrainingInterface");

async function readQueue(inputQueue, waitTimeThreshold) {
    return new Promise(async function (resolve, reject) {
        try {
            const connection = await amqp.connect(queueUrl);
            const channel = await connection.createChannel();

            process.once("SIGINT", async () => {
                await channel.close();
                await connection.close();
                resolve(null);
            });
            await channel.assertQueue(inputQueue, { durable: true });
            await channel.prefetch(1);
            if (waitTimeThreshold) {
                var timeoutInterval = setTimeout(async () => {
                    await channel.close();
                    await connection.close();
                    resolve(null);
                }, waitTimeThreshold);
            }
            await channel.consume(
                inputQueue,
                async (message) => {
                    if (waitTimeThreshold && timeoutInterval) {
                        clearInterval(timeoutInterval);
                    }
                    if (message) {
                        resolve(JSON.parse((message.content).toString()));
                    }
                    else{
                        resolve(null);
                    }
                    await channel.ack(message);
                    await channel.close();
                    await connection.close();
                }
            );

        } catch (err) {
            reject(err);
        }
    });
}


async function writeQueue(outputQueue, tfjsJobResponse) {
    //console.log(`writeQueue outputQueue: ${outputQueue}, tfjsJobResponse: ${JSON.stringify(tfjsJobResponse)}`)
    return new Promise(async function (resolve, reject) {
        let connection;
        try {
            connection = await amqp.connect(queueUrl);
            const channel = await connection.createChannel();

            await channel.assertQueue(outputQueue, { durable: true });
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

module.exports = class WorkerTraining extends DistributedTrainingInterface {
    constructor(jobName, parallelism, podResponseTimeThreshold, alternativeWorker) {
        super();
        jobName = "tfjsjob"
        this.jobName = jobName;
        this.outputQueuePrefix = jobName + "-OUTPUT";
        this.inputQueue = jobName + "-INPUT";
        this.parallelism = parallelism;
        this.podResponseTimeThreshold = podResponseTimeThreshold;
        this.alternativeWorker = alternativeWorker;
        console.log(`jobName ${jobName}`);
    }

    trainModel(phenotype, modelJson, tensors, validationSplit, modelAbortThreshold, modelTrainingTimeThreshold) {
        var self = this;
        console.log(`trainModel -> jobName: ${self.jobName}, phenotype._id: ${phenotype._id}`);
        
        return new Promise((resolve, reject) => {
            var runTask = async function (resolve, reject) {
                try {
                    
                    await writeQueue(self.inputQueue, {
                        workerData: { 
                            phenotype: phenotype, 
                            modelJson : modelJson, 
                            tensors: tensors,
                            validationSplit: validationSplit, 
                            modelAbortThreshold: modelAbortThreshold,
                            modelTrainingTimeThreshold: modelTrainingTimeThreshold
                        },
                    });
                    var tfjsJob = (await readQueue(`${self.outputQueuePrefix}-${phenotype._id}`, self.podResponseTimeThreshold * 1000));
                    if (tfjsJob && tfjsJob.modelJson) {
                        switch (self.alternativeWorker) {
                            case "python":
                                {
                                    ModelStorage.writeModelBuffer(phenotype._id, Buffer.from(tfjsJob.modelJson, "hex"));
                                }
                                break;
                            default:
                                {
                                    const modelData = JSON.parse(tfjsJob.modelJson);
                                    const weightData = new Uint8Array(Buffer.from(modelData.weightData, "base64")).buffer;
                                    const model = await tf.loadLayersModel(tf.io.fromMemory(
                                        {
                                            modelTopology: modelData.modelTopology,
                                            weightSpecs: modelData.weightSpecs,
                                            weightData: weightData

                                        }));
                                    ModelStorage.writeModel(phenotype._id, model);
                                }
                                break;
                        }
                        resolve({ validationLoss: tfjsJob.validationLoss, phenotype: tfjsJob.phenotype });
                    } else {
                        resolve({ validationLoss: NaN, phenotype: phenotype });
                    } 
                }
                catch (e) { reject(e) }                    
            }            
            runTask(resolve, reject);
        })
    }
}