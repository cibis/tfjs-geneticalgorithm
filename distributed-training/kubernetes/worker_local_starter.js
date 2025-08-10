const amqp = require("amqplib");
const tf = require('@tensorflow/tfjs-node');
var ModelStorage = require("../../model-storage/current")

const queueUrl = "amqp://192.168.2.28:30000";

const rabbitmqDefaultConnectionSettings = {
    hostname: '192.168.2.28',
    port: 30000,
    username: 'guest',
    password: 'guest',
    protocol: 'amqp',
    locale: 'en_US',
    frameMax: 0, 
    heartbeat: 0,
    vhost: '/'    
};

var DistributedTrainingInterface = require("../DistributedTrainingInterface");

async function readQueue(inputQueue, waitTimeThreshold, rabbitmqConnectionSettings) {
    return new Promise(async function (resolve, reject) {
        try {
            //console.log(`${Date.now()} readQueue ${inputQueue}`);
            const connection = await amqp.connect(Object.assign(rabbitmqDefaultConnectionSettings, rabbitmqConnectionSettings));
            const channel = await connection.createChannel();

            process.once("SIGINT", async () => {
                try {
                    await channel.close();
                    await connection.close();
                }
                catch (closeErr) { }
                resolve(null);
            });
            //console.log(`RESULT QUEUE ${inputQueue}`)
            await channel.assertQueue(inputQueue, { durable: true });
            await channel.prefetch(1);
            if (waitTimeThreshold) {
                var timeoutInterval = setTimeout(async () => {
                    try {
                        await channel.close();
                        await connection.close();
                    }
                    catch (closeErr) { }
                    resolve(null);
                }, waitTimeThreshold + (10 * 60 * 1000)/* the extra time required for caching the dataset on the worker*/);
            }
            await channel.consume(
                inputQueue,
                async (message) => {
                    if (waitTimeThreshold && timeoutInterval) {
                        clearInterval(timeoutInterval);
                    }
                    if (message) {
                        var jsonParse = null;
                        try {
                            jsonParse = JSON.parse((message.content).toString());
                        }
                        catch (parseErr) {
                            console.log(`worker response parse error: err: ${parseErr}`)
                        }
                        resolve(jsonParse);
                    }
                    else{
                        resolve(null);
                    }
                    await channel.ack(message);
                    try {
                        await channel.close();
                        await connection.close();
                    }
                    catch (closeErr) { }
                }
            );

        } catch (err) {
            reject(err);
        }
    });
}


async function writeQueue(outputQueue, tfjsJobResponse, rabbitmqConnectionSettings) {
    return new Promise(async function (resolve, reject) {
        let connection;
        try {
            connection = await amqp.connect(Object.assign(rabbitmqDefaultConnectionSettings, rabbitmqConnectionSettings));
            const channel = await connection.createChannel();

            await channel.assertQueue(outputQueue, { durable: true });
            channel.sendToQueue(outputQueue, Buffer.from(JSON.stringify(tfjsJobResponse)));
            try {
                await channel.close();
                await connection.close();
            }
            catch (closeErr) { }
            resolve();
        } catch (err) {
            console.warn(err);
            reject(err);
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

    async startJob() {
        
    }

    async stopJob() {

    }

    trainModel(phenotype, modelJson, tensors, validationSplit, modelAbortThreshold, modelTrainingTimeThreshold, baseline) {
        var self = this;
        console.log(`trainModel -> jobName: ${self.jobName}, phenotype._id: ${phenotype._id}`);
        
        return new Promise((resolve, reject) => {
            var runTask = async function (resolve, reject) {
                try {
                    console.log(`${new Date().toLocaleTimeString()} send to worker: ${JSON.stringify(phenotype)}\n`);
                    await writeQueue(self.inputQueue, {
                        workerData: { 
                            phenotype: phenotype, 
                            modelJson : modelJson, 
                            tensors: tensors,
                            validationSplit: validationSplit, 
                            modelAbortThreshold: modelAbortThreshold,
                            modelTrainingTimeThreshold: modelTrainingTimeThreshold,
                            baseline: baseline
                        },
                    });
                    var tfjsJob = (await readQueue(`${self.outputQueuePrefix}-${phenotype._id}`, self.podResponseTimeThreshold * 1000));
                    try {
                        if (tfjsJob && tfjsJob.modelJson) {
                            switch (self.alternativeWorker) {
                                case "python":
                                    {
                                        ModelStorage.writeModelBuffer(phenotype._id, Buffer.from(tfjsJob.modelJson, "hex"), tfjsJob.phenotype);
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
                            resolve({ validationLoss: tfjsJob.validationLoss, phenotype: tfjsJob.phenotype, predictionsJson: tfjsJob.predictionsJson });
                        } else {
                            resolve({ validationLoss: NaN, phenotype: phenotype });
                        }
                    }
                    catch (readQueueEx) { 
                        resolve({ validationLoss: NaN, phenotype: phenotype });
                    }
                }
                catch (e) { reject(e) }                    
            }            
            runTask(resolve, reject);
        })
    }
}