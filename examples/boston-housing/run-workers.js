const tf = require('@tensorflow/tfjs-node');
const BostonHousing = require('./boston-housing')
var WorkerTraining = require("../../distributed-training/workers/worker_starter")
var TFJSGeneticAlgorithmConstructor = require("../../index")
var ModelStorage = require("../../model-storage/current")

async function testPredefinedModelsAgainstGA() {
    var bestPredefinedModelLoss = await BostonHousing.runPredefinedModels();
    var bostonData = BostonHousing.getBostonData();
    console.log(`Best predefined models validation-set loss ${bestPredefinedModelLoss}\n`)

    var taskSettings = {
        parallelism: 12,
        modelTrainingTimeThreshold: (60 * 2)/* 2 min */,
        populationSize: 200,
        baseline: 85,
        predefinedModelCloneCompetitionSize: 100,
        evolveGenerations: 2,
        elitesGenerations: 2,
        finalCloneCompetitionSize: 10,
    };

    var ga = TFJSGeneticAlgorithmConstructor({
        parallelProcessing: true,
        parallelism: taskSettings.parallelism,
        modelTrainingTimeThreshold: taskSettings.modelTrainingTimeThreshold,
        modelTrainingFuction: async function (phenotype, model) {
            try {
                var worker = new WorkerTraining();

                await ModelStorage.writeModel(phenotype._id, model);

                var workerResponse = await worker.trainModel(phenotype, this.tensors, this.validationSplit, this.modelAbortThreshold, this.modelTrainingTimeThreshold);
                phenotype.epochs = workerResponse.phenotype.epochs;
                return { validationLoss: workerResponse.validationLoss }
            }
            catch (err) {
                console.log(`Error: ${err} stack: ${err.stack}`)
                throw err;
            }
        },        
        populationSize: taskSettings.populationSize,
        baseline: taskSettings.baseline,
        tensors: BostonHousing.getTensor(),
        parameterMutationFunction: (oldPhenotype) => {
            if (!oldPhenotype) {
                return {
                    epochs: 200,
                    batchSize: 40,
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
                    epochs: ga.mutateNumber(oldPhenotype.epochs, true, 200, true, 10),
                    batchSize: ga.mutateNumber(oldPhenotype.batchSize, true, 200, true, 10),
                    learningRate: ga.mutateNumber(oldPhenotype.learningRate, false, 200, true),
                    hiddenLayers: ga.mutateNumber(oldPhenotype.hiddenLayers, true, 200, false, 0, 5),
                    hiddenLayerUnits: ga.mutateNumber(oldPhenotype.hiddenLayerUnits, true, 200, true, 10, 500),
                    activation: ga.mutateOptions(oldPhenotype.activation, ga.ACTIVATIONS),
                    kernelInitializer: ga.mutateOptions(oldPhenotype.kernelInitializer, ga.KERNEL_INITIALIZERS),
                    optimizer: ga.mutateOptions(oldPhenotype.optimizer, ga.OPTIMIZERS),
                    loss: 'meanSquaredError'
                };
        },  
        modelBuilderFunction: (phenotype) => {
            const model = tf.sequential();
            if (phenotype.hiddenLayers > 0) {
                model.add(tf.layers.dense({
                    inputShape: [bostonData.numFeatures],
                    units: phenotype.hiddenLayerUnits,
                    activation: phenotype.activation,
                    kernelInitializer: phenotype.kernelInitializer
                }));

                for (var i = 1; i < phenotype.hiddenLayers; i++) {
                    model.add(tf.layers.dense(
                        { units: phenotype.hiddenLayerUnits, activation: phenotype.activation, kernelInitializer: phenotype.kernelInitializer }));
                }
                model.add(tf.layers.dense({ units: 1 }));
            }
            else {
                model.add(tf.layers.dense({ inputShape: [bostonData.numFeatures], units: 1 }));
            }
            return model;
        }
    });
    console.log(`Running cloneCompete for predefined model`)
    var cloneCompetePredefinedModel = await ga.cloneCompete({
        epochs: 200,
        batchSize: 40,
        learningRate: 0.01,
        hiddenLayers: 2,
        hiddenLayerUnits: 50,
        activation: 'sigmoid',
        kernelInitializer: 'leCunNormal',
        optimizer: 'sgd',
        loss: 'meanSquaredError'
    }, taskSettings.predefinedModelCloneCompetitionSize);
    console.log(`Best loss of all predefined model after cloneCompete`)
    console.log(cloneCompetePredefinedModel.validationLoss);

    console.log("\n\nTraining/Testing models with predefined structure.\n")
    var bestModel = await ga.evolve(taskSettings.evolveGenerations,taskSettings.elitesGenerations);
    console.log(`Best of all GA model parameters`)
    console.log(bestModel);
    bestModel = await ga.cloneCompete(bestModel, taskSettings.finalCloneCompetitionSize);
    console.log(`Best of all GA model after cloneCompete`)
    console.log(bestModel); 
    ModelStorage.copyToBest(bestModel, "boston-housing");
    console.log(`Best predefined models loss after cloneCompete ${cloneCompetePredefinedModel.validationLoss}`)
    console.log(`Best GA models loss after cloneCompete ${bestModel.validationLoss}`) 
    if(cloneCompetePredefinedModel.validationLoss > bestModel.validationLoss){
        console.log("Genetic Algorithm WON!!!");
    }
    else{
        console.log("Genetic Algorithm lost :( ");
    }
}

testPredefinedModelsAgainstGA();