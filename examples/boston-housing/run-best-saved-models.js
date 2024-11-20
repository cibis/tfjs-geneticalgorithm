const tf = require('@tensorflow/tfjs-node');
const BostonHousing = require('./boston-housing')
var ModelStorage = require("../../model-storage/current")
var utils = require("../../utils")

const BATCH_SIZE = 40;

async function main() {
    var betsModels = ModelStorage.listBestModels();
    await BostonHousing.runPredefinedModels();
    var tensors = BostonHousing.getTensor();
    console.log("\n\nBEST SAVED MODELS:");
    for (var i = 0; i < betsModels.length; i++) {    
        var model = await ModelStorage.readBestModel(betsModels[i]);
        if(model.phenotype.group != "boston-housing") continue;
        console.log(model.phenotype);
        model.compile(
            { optimizer: utils.optimizerBuilderFunction(model.phenotype.optimizer, model.phenotype.learningRate), loss: model.phenotype.loss });

        const result = model.evaluate(
            tensors.testFeatures, tensors.testTarget, { batchSize: BATCH_SIZE });
        const testLoss = result.dataSync()[0];
        console.log(
            `Test-set loss: ${testLoss.toFixed(4)}`);
        console.log("============================================================\n");
    }
}

main();
