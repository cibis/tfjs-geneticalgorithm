const amqp = require("amqplib");
const k8s = require('@kubernetes/client-node');
const tf = require('@tensorflow/tfjs-node');
var ModelStorage = require("../../model-storage/current")

const NEW_MESSAGE_WAIT_TIME = 20000;

//kubectl port-forward service/rabbitmq-service 30000:5672
const queueUrl = "amqp://127.0.0.1:30000";

const kc = new k8s.KubeConfig();
kc.loadFromDefault();

const k8sApi = kc.makeApiClient(k8s.CoreV1Api);
const k8sBatchApi = kc.makeApiClient(k8s.BatchV1Api);

var DistributedTrainingInterface = require("../DistributedTrainingInterface");
var utils = require("../../utils")

async function readQueue(inputQueue, waitTimeThreshold) {
    return new Promise(async function (resolve, reject) {
        try {
            //console.log(`${Date.now()} readQueue ${inputQueue}`);
            const connection = await amqp.connect(queueUrl);
            const channel = await connection.createChannel();

            process.once("SIGINT", async () => {
                await channel.close();
                await connection.close();
                resolve(null);
            });

            await channel.assertQueue(inputQueue, { durable: false });
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
                        resolve(JSON.parse(message.content.toString()));
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
    return new Promise(async function (resolve, reject) {
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

const deleteJob = async (jobName) => {
    await k8sBatchApi.deleteNamespacedJob(jobName, 'default', propagationPolicy = 'Background');
    (await k8sApi.listNamespacedPod('default')).body.items.forEach(async pod => {
        if (pod.metadata.labels["job-name"] == jobName) {
            await k8sApi.deleteNamespacedPod(pod.metadata.name, 'default');
        }
    })
}

const isJobStarted = async (jobName) => {
    return !!(await k8sBatchApi.listNamespacedJob('default')).body.items.find(o=>o.metadata.name == jobName);
}


const createJob = async (jobName, parallelism) => {
    await k8sBatchApi.createNamespacedJob('default', {
        apiVersion: 'batch/v1',
        kind: 'Job',
        metadata: {
            name: jobName
        },
        spec: {
            parallelism: parallelism,
            template: {
                metadata: {
                    name: 'tfjs-ks-worker'
                },
                spec: {
                    volumes: [
                        {
                            name: "job-tfjs-node-storage",
                            persistentVolumeClaim: {
                                claimName: "job-tfjs-node-claim"
                            }
                        }
                    ],
                    containers: [{
                        image: 'tfjs-ks-worker:latest',
                        name: 'tfjs-ks-worker',
                        imagePullPolicy: 'Never',
                        volumeMounts: [
                            {
                                mountPath: "/shared",
                                name: "job-tfjs-node-storage"
                            }
                        ],
                        env: [
                            {
                                name: "JOB_NAME",
                                value: jobName
                            }
                        ]
                    }],
                    restartPolicy: "Never"
                }
            }
        }
    });

}

module.exports = class WorkerTraining extends DistributedTrainingInterface {
    constructor(jobName, parallelism, podResponseTimeThreshold) {
        super();
        this.jobName = jobName;
        this.outputQueuePrefix = jobName + "-OUTPUT";
        this.inputQueue = jobName + "-INPUT";
        this.parallelism = parallelism;
        this.podResponseTimeThreshold = podResponseTimeThreshold;
        console.log(`jobName ${jobName}`);
    }

    async startJob() {
        if(!(await isJobStarted(this.jobName))) await createJob(this.jobName, this.parallelism);
    }

    async stopJob() {
        await deleteJob(this.jobName);
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
                    var tfjsJob = (await readQueue(`${self.outputQueuePrefix}-${phenotype._id}`, self.podResponseTimeThreshold * 1000));
                    if (tfjsJob && tfjsJob.modelJson) {
                        const modelData = JSON.parse(tfjsJob.modelJson);
                        const weightData = new Uint8Array(Buffer.from(modelData.weightData, "base64")).buffer;
                        const model = await tf.loadLayersModel(tf.io.fromMemory(
                            { 
                                modelTopology: modelData.modelTopology, 
                                weightSpecs: modelData.weightSpecs, 
                                weightData: weightData 

                            }));
                        ModelStorage.writeModel(phenotype._id, model);
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