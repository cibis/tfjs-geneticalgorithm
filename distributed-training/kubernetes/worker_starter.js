const amqp = require("amqplib");
const k8s = require('@kubernetes/client-node');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
var ModelStorage = require("../../model-storage/current")

const NEW_MESSAGE_WAIT_TIME = 20000;

//kubectl port-forward service/rabbitmq-service 30000:5672
const rabbitmqDefaultConnectionSettings = {
    hostname: 'localhost',
    port: 30000,
    username: 'guest',
    password: 'guest',
    protocol: 'amqp',
    locale: 'en_US',
    frameMax: 0, 
    heartbeat: 0,
    vhost: '/'    
};

var k8sApi;
var k8sBatchApi;

function initKB(data) {
    const kc = new k8s.KubeConfig();
    if (!data)
        kc.loadFromDefault();
    else
        kc.loadFromString(data);
    //const data = fs.readFileSync('kb-config.yaml', 'utf8');
    
    k8sApi = kc.makeApiClient(k8s.CoreV1Api);
    k8sBatchApi = kc.makeApiClient(k8s.BatchV1Api);
}


var DistributedTrainingInterface = require("../DistributedTrainingInterface");
var utils = require("../../utils")

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
                }, waitTimeThreshold);
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

const deleteJob = async (jobName) => {
    await k8sBatchApi.deleteNamespacedJob( name = jobName, namespace = 'default', propagationPolicy = 'Foreground');
    (await k8sApi.listNamespacedPod('default')).body.items.forEach(async pod => {
        if (pod.metadata.labels["job-name"] == jobName) {
            console.log(`deleting pod ${pod.metadata.name}`);
            k8sApi.deleteNamespacedPod( name = pod.metadata.name, namespace = 'default', gracePeriodSeconds = 0 );
        }
    })
}

const isJobStarted = async (jobName) => {
    return !!(await k8sBatchApi.listNamespacedJob('default')).body.items.find(o=>o.metadata.name == jobName);
}


const createJob = async (jobName, parallelism, alternativeWorker, resources, env) => {
    console.log(`createJob alternativeWorker: ${alternativeWorker}`)
    var imageName = typeof alternativeWorker != "undefined" && alternativeWorker == "python" ? 'tfjs-ks-python-worker' : 'tfjs-ks-worker'
    var jobSettings = {
        apiVersion: 'batch/v1',
        kind: 'Job',
        metadata: {
            name: jobName
        },
        spec: {
            parallelism: parallelism,
            template: {
                metadata: {
                    name: imageName
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
                        image: imageName + ':latest',
                        name: imageName,
                        imagePullPolicy: 'Never',
                        volumeMounts: [
                            {
                                mountPath: "/shared",
                                name: "job-tfjs-node-storage"
                            }
                        ],
                        env: [...[
                            {
                                name: "JOB_NAME",
                                value: jobName
                            }
                        ], ...env],
                        resources: resources
                    }],
                    restartPolicy: "Never"
                }
            }
        }
    };
    console.log(JSON.stringify(jobSettings))
    await k8sBatchApi.createNamespacedJob('default', jobSettings);

}

module.exports = class WorkerTraining extends DistributedTrainingInterface {
    constructor(jobName, parallelism, podResponseTimeThreshold, alternativeWorker, settings) {
        super();
        this.jobName = jobName;
        this.outputQueuePrefix = jobName + "-OUTPUT";
        this.inputQueue = jobName + "-INPUT";
        this.parallelism = parallelism;
        this.podResponseTimeThreshold = podResponseTimeThreshold;
        this.alternativeWorker = alternativeWorker;
        this.settings = settings;  
        this.resources = {};    
        this.env = [];  
        if (settings) {
            this.rabbitmqConnectionSettings = settings.rabbitmqConnectionSettings;
            if(settings.resources) this.resources = settings.resources;
            if(settings.env && settings.env.length) this.env = settings.env;
            if(settings.kb)
                this.kb = settings.kb;
        } 
        initKB(settings.kb);
        console.log(`jobName ${jobName}`);
    }

    async startJob() {
        if(!(await isJobStarted(this.jobName))) await createJob(this.jobName, this.parallelism, this.alternativeWorker, this.resources, this.env);
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
                    console.log(`${new Date().toLocaleTimeString()} send to worker: ${JSON.stringify(phenotype)}\n`);
                    await writeQueue(self.inputQueue, {
                        workerData: { 
                            phenotype: phenotype, 
                            modelJson : modelJson, 
                            tensors: tensors,
                            validationSplit: validationSplit, 
                            modelAbortThreshold: modelAbortThreshold,
                            modelTrainingTimeThreshold: modelTrainingTimeThreshold
                        },
                    }, self.rabbitmqConnectionSettings);
                    var tfjsJob = (await readQueue(`${self.outputQueuePrefix}-${phenotype._id}`, self.podResponseTimeThreshold * 1000, self.rabbitmqConnectionSettings));
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
                                    ModelStorage.writeModel(phenotype._id, model, tfjsJob.phenotype);
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