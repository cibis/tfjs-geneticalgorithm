//require('../../utils').initConsoleLogTimestamps();
const tf = require('@tensorflow/tfjs-node');
const JenaWeather = require('./jena-weather')
var TFJSGeneticAlgorithmConstructor = require("../../index")
var ModelStorage = require("../../model-storage/current")
var ExampleDataService = require('../example-data-service');
var DataService = require('../../data-service');

async function testPredefinedModelsAgainstGA() {

    console.log('Loading Jena weather data (41.2 MB)...');
    jenaWeatherData = new JenaWeather.JenaWeatherData();
    await jenaWeatherData.load();
    let numFeatures = jenaWeatherData.getDataColumnNames().length;
    const lookBack = 10 * 24 * 6;  // Look back 10 days.
    const step = 6;   
    const inputShape = [Math.floor(lookBack / step), numFeatures];
    console.log('Done loading Jena weather data.');

    var bestPredefinedModelLoss =/* 1 ?? */await JenaWeather.runPredefinedModels(jenaWeatherData);
    console.log(`Best predefined models validation-set loss ${bestPredefinedModelLoss}\n`)

    await ExampleDataService.load();

    var taskSettings = {
        populationSize: 20,
        baseline: 24,
        predefinedModelCloneCompetitionSize: 10,
        evolveGenerations: 5,
        elitesGenerations: 2,
        finalCloneCompetitionSize: 10,
    };

    var ga = TFJSGeneticAlgorithmConstructor({
        /*
        //use for distributed/parallel training
        parallelProcessing: true,
        modelTrainingFuction: async () =>{

        },
        */
        populationSize: taskSettings.populationSize,
        baseline: taskSettings.baseline,
        tensors: new DataService.DataSetSources(
            new DataService.DataSetSource("127.0.0.1", "/jena-weather-training", "3000", "jena-weather-training", 1280),
            new DataService.DataSetSource("127.0.0.1", "/jena-weather-validation", "3000", "jena-weather-validation", 1280)
        ),//BostonHousing.getTensor(),
        batchesPerEpoch: 500,
        parameterMutationFunction: (oldPhenotype) => {
            if (!oldPhenotype) {
                return {
                    epochs: 1,
                    batchSize: 128,
                    learningRate: 0.01,
                    hiddenLayers: 0,
                    hiddenLayerUnits: 50,
                    activation: 'sigmoid',
                    kernelInitializer: 'leCunNormal',
                    optimizer: 'sgd',
                    loss: 'meanSquaredError'
                };
            }
            else
                return {
                    epochs: ga.mutateNumber(oldPhenotype.epochs, true, 50, true, 70),
                    batchSize: ga.mutateNumber(oldPhenotype.batchSize, true, 50, true, 10),
                    learningRate: ga.mutateNumber(oldPhenotype.learningRate, false, 100, true),
                    hiddenLayers: ga.mutateNumber(oldPhenotype.hiddenLayers, true, 200, false, 0, 10),
                    hiddenLayerUnits: ga.mutateNumber(oldPhenotype.hiddenLayerUnits, true, 100, true, 1, 300),
                    activation: ga.mutateOptions(oldPhenotype.activation, ga.ACTIVATIONS),
                    kernelInitializer: ga.mutateOptions(oldPhenotype.kernelInitializer, ga.KERNEL_INITIALIZERS),
                    optimizer: ga.mutateOptions(oldPhenotype.optimizer, ga.OPTIMIZERS),
                    loss: 'meanSquaredError'
                };
        },
        modelBuilderFunction: (phenotype) => {
            const model = tf.sequential();
            model.add(tf.layers.flatten({ inputShape }));
            if (phenotype.hiddenLayers > 0) {
                model.add(tf.layers.dense({
                    //inputShape: inputShape,
                    units: phenotype.hiddenLayerUnits,
                    activation: phenotype.activation,
                    //kernelInitializer: phenotype.kernelInitializer
                }));

                for (var i = 1; i < phenotype.hiddenLayers; i++) {
                    model.add(tf.layers.dense(
                        { units: phenotype.hiddenLayerUnits, activation: phenotype.activation, kernelInitializer: phenotype.kernelInitializer }));
                }
                model.add(tf.layers.dense({ units: 1 }));
            }
            else {
                model.add(tf.layers.dense({ units: 1 }));
            }
            model.compile(
                { optimizer: ga.optimizerBuilderFunction(phenotype.optimizer, phenotype.learningRate), loss: phenotype.loss });
            //model.summary();
            
            return model;
        }
    });

    console.log("\n\nTraining/Testing models with predefined structure.\n")
    var bestModel = await ga.evolve(taskSettings.evolveGenerations, taskSettings.elitesGenerations);
    console.log(`Best of all GA model parameters`)
    console.log(bestModel);
    bestModel = await ga.cloneCompete(bestModel, taskSettings.finalCloneCompetitionSize);
    console.log(`Best of all GA model after cloneCompete`)
    console.log(bestModel);
    ModelStorage.copyToBest(bestModel, "boston-housing");
    console.log(`Best predefined models loss ${bestPredefinedModelLoss}`)
    console.log(`Best GA models loss after cloneCompete ${bestModel.validationLoss}`)
    if (bestPredefinedModelLoss > bestModel.validationLoss) {
        console.log("Genetic Algorithm WON!!!");
    }
    else {
        console.log("Genetic Algorithm lost :( ");
    }
    process.exit(0);
}

testPredefinedModelsAgainstGA();