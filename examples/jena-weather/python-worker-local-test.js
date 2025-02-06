/*
run this:
tfjs-geneticalgorithm\distributed-training\kubernetes\docker\python_worker> python worker-local-test.py

before running:
tfjs-geneticalgorithm> node examples\jena-weather\python-worker-local-test.js
*/


const tf = require('@tensorflow/tfjs-node');
const fs = require('fs')
const JenaWeather = require('./jena-weather')
var WorkerTraining = require("../../distributed-training/kubernetes/worker_local_starter");
var TFJSGeneticAlgorithmConstructor = require("../../index")
var ModelStorage = require("../../model-storage/current")
var ExampleDataService = require('../example-data-service');
var DataService = require('../../data-service');
var utils = require("../../utils")

async function testPredefinedModelsAgainstGA() {
    console.log('Loading Jena weather data (41.2 MB)...');
    jenaWeatherData = new JenaWeather.JenaWeatherData();
    await jenaWeatherData.load();
    let numFeatures = jenaWeatherData.getDataColumnNames().length;
    const lookBack = 10 * 24 * 6;  // Look back 10 days.
    const step = 6;
    const inputShape = [Math.floor(lookBack / step), numFeatures];
    console.log('Done loading Jena weather data.');

    var bestPredefinedModelLoss =
        0.2797 ?? //comment this line if you want to run the jena weather example models
        await JenaWeather.runPredefinedModels(jenaWeatherData);
    console.log(`Best predefined models validation-set loss ${bestPredefinedModelLoss}\n`)

    await ExampleDataService.load();

    var taskSettings = {
        parallelism: 1,
        //calculate in advance based on first epoch time
        modelTrainingTimeThreshold: (60 * 60 * 2)/* 2 h */,
        populationSize: 30,
        baseline: 24,
        evolveGenerations: 5,
        elitesGenerations: 2,
        finalCloneCompetitionSize: 10,
    };

    var worker = new WorkerTraining(`job-tfjs-node-${utils.guidGenerator()}`, taskSettings.parallelism, taskSettings.modelTrainingTimeThreshold * 3, "python");
    try {
        var ga = TFJSGeneticAlgorithmConstructor({
            parallelProcessing: true,
            parallelism: taskSettings.parallelism,
            modelTrainingTimeThreshold: taskSettings.modelTrainingTimeThreshold,
            modelTrainingFuction: async function (phenotype, model) {
                try {
                    const modelJson = {
                        buildModel: fs.readFileSync('./examples/jena-weather/buildModel.py').toString()
                    };

                    var workerResponse = await worker.trainModel(phenotype, modelJson, this.tensors, this.validationSplit, this.modelAbortThreshold, this.modelTrainingTimeThreshold);
                    phenotype.epochs = workerResponse.phenotype.epochs;
                    console.log(`Model training completed ${phenotype._id} . loss ${workerResponse.validationLoss}`);
                    return { validationLoss: parseFloat(workerResponse.validationLoss) }
                }
                catch (err) {
                    console.log(`Error: ${err} stack: ${err.stack}`)
                    throw err;
                }
            },
            populationSize: taskSettings.populationSize,
            baseline: taskSettings.baseline,
            tensors: new DataService.DataSetSources(
                new DataService.DataSetSource("127.0.0.1", "/jena-weather-training", "3000", "jena-weather-training", 1280),
                new DataService.DataSetSource("127.0.0.1", "/jena-weather-validation", "3000", "jena-weather-validation", 1280)
            ),
            //tensors: BostonHousing.getTensor(),
            parameterMutationFunction: (oldPhenotype) => {
                if (!oldPhenotype) {
                    return {
                        epochs: 1,
                        batchSize: 128,
                        learningRate: 0.01,
                        hiddenLayers: 1,
                        hiddenLayerUnits: 50,
                        activation: 'sigmoid',
                        kernelInitializer: 'leCunNormal',
                        optimizer: 'sgd',
                        loss: 'meanAbsoluteError',
                        modelType: "linear",
                        add_dropout: false,
                        dropoutRate: 0,
                        recurrentDropout: 0
                    };
                }
                else {
                    var newPhenotype = {
                        epochs: ga.mutateNumber(oldPhenotype.epochs, true, 50, true, 5),
                        batchSize: ga.mutateNumber(oldPhenotype.batchSize, true, 50, true, 10),
                        learningRate: ga.mutateNumber(oldPhenotype.learningRate, false, 100, true),
                        hiddenLayerUnits: ga.mutateNumber(oldPhenotype.hiddenLayerUnits, true, 100, true, 1, 300),
                        activation: ga.mutateOptions(oldPhenotype.activation, ga.ACTIVATIONS),
                        kernelInitializer: ga.mutateOptions(oldPhenotype.kernelInitializer, ga.KERNEL_INITIALIZERS),
                        optimizer: ga.mutateOptions(oldPhenotype.optimizer, ga.OPTIMIZERS),
                        loss: 'meanAbsoluteError',
                        modelType: ga.mutateOptions(oldPhenotype.modelType, ["linear", "mlp", "simpleRNN", "gru", "lstm"]),
                        add_dropout: false,
                        dropoutRate: 0,
                        recurrentDropout: 0
                    };
    
                    switch (newPhenotype.modelType) {
                        case "linear":
                            newPhenotype.hiddenLayers = 0;
                            break;
                        case "mlp":
                            newPhenotype.hiddenLayers = ga.mutateNumber(oldPhenotype.hiddenLayers, true, 200, false, 1, 10);
                            if (newPhenotype.add_dropout) {
                                newPhenotype.dropoutRate = ga.mutateNumber(newPhenotype.dropoutRate, false, 50, false, 0, 1);
                            }
                            break;
                        case "simpleRNN":
                        case "gru":
                        case "lstm":
                            newPhenotype.hiddenLayers = 1;
                            if (newPhenotype.add_dropout) {
                                newPhenotype.recurrentDropout = ga.mutateNumber(newPhenotype.recurrentDropout, false, 50, false, 0, 1);
                            }
                            break;
                    }
                    return newPhenotype;
                }
            },
            modelBuilderFunction: (phenotype) => {                
                return { message: "The model building will happen inside the worker using buildModel.py -> buildModel" };
            }
        });
        
        
        console.log("\n\nTraining/Testing models with predefined structure.\n")
        var bestModel = await ga.evolve(taskSettings.evolveGenerations, taskSettings.elitesGenerations);
        console.log(`Best of all GA model parameters`)
        console.log(bestModel);
        bestModel = await ga.cloneCompete(bestModel, taskSettings.finalCloneCompetitionSize);
        console.log(`Best of all GA model after cloneCompete`)
        console.log(bestModel);
        //ModelStorage.copyToBest(bestModel, "jena-weather");
        console.log(`Best predefined models loss ${bestPredefinedModelLoss}`)
        console.log(`Best GA models loss after cloneCompete ${bestModel.validationLoss}`)
        if (bestPredefinedModelLoss > bestModel.validationLoss) {
            console.log("Genetic Algorithm WON!!!");
        }
        else {
            console.log("Genetic Algorithm lost :( ");
        }
    }
    catch (err) {
        console.log(`Error: ${err} stack: ${err.stack}`)
    }
    finally {
        console.log(`${Date.now()} worker.stopJob`)
    }
    process.exit(0);
}

testPredefinedModelsAgainstGA();