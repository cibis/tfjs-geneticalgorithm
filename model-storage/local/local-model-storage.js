const tf = require('@tensorflow/tfjs-node');
var fs = require('fs');
var ModelStorageInterface = require("../ModelStorageInterface");
const BEST_MODEL_STORAGE = "./_runtime/best/"
const MODEL_STORAGE = "./_runtime/models/"

module.exports = class LocalModelStorage extends ModelStorageInterface {
    constructor() {
        super();
    }

    async readModel(modelId) {
        //console.log(`${Date.now()} LocalModelStorage readModel ${modelId}`);
        var model = await tf.loadLayersModel(`file://${MODEL_STORAGE}${modelId}/model.json`);
        return model;
    }

    async writeModel(modelId, model, phenotype){
        if(!modelId) throw Error("modelId is undefined");
        if (!fs.existsSync(BEST_MODEL_STORAGE)){
            fs.mkdirSync(BEST_MODEL_STORAGE, { recursive: true });
        }
        if (!fs.existsSync(MODEL_STORAGE)){
            fs.mkdirSync(MODEL_STORAGE, { recursive: true });
        }
        //console.log(`${Date.now()} LocalModelStorage writeModel ${modelId} ${!!model}`);
        await model.save(`file://${MODEL_STORAGE}${modelId}`);
        if (phenotype)
            fs.writeFileSync(`${MODEL_STORAGE}${modelId}/phenotype.json`, JSON.stringify(phenotype), 'utf8')
    }

    async writePhenotype(modelId, phenotype){
        if(!modelId) throw Error("modelId is undefined");
        fs.writeFileSync(`${MODEL_STORAGE}${modelId}/phenotype.json`, JSON.stringify(phenotype), 'utf8')
    }    

    async writeModelBuffer(modelId, bufferData, phenotype){
        if(!modelId) throw Error("modelId is undefined");
        if (!fs.existsSync(BEST_MODEL_STORAGE)){
            fs.mkdirSync(BEST_MODEL_STORAGE, { recursive: true });
        }
        if (!fs.existsSync(`${MODEL_STORAGE}${modelId}/`)){
            fs.mkdirSync(`${MODEL_STORAGE}${modelId}/`, { recursive: true });
        }
        fs.writeFileSync(`${MODEL_STORAGE}${modelId}/model.keras`, bufferData);
        fs.writeFileSync(`${MODEL_STORAGE}${modelId}/phenotype.json`, JSON.stringify(phenotype), 'utf8')
    }

    copyToBest(phenotype, group){
        phenotype.group = group;
        fs.writeFileSync(`${BEST_MODEL_STORAGE}${phenotype._id}.json`, JSON.stringify(phenotype), 'utf8'); 
        fs.cpSync(MODEL_STORAGE + phenotype._id, BEST_MODEL_STORAGE + phenotype._id, {recursive: true});
    }

    listBestModels(){
        const getDirectories = source =>
            fs.readdirSync(source, { withFileTypes: true })
              .filter(dirent => dirent.isDirectory())
              .map(dirent => dirent.name)
        return getDirectories(BEST_MODEL_STORAGE);
    }
    async readBestModel(modelId) {
        //console.log(`${Date.now()} LocalModelStorage readModel ${modelId}`);
        var model = await tf.loadLayersModel(`file://${BEST_MODEL_STORAGE}${modelId}/model.json`);
        var phenotype = JSON.parse((fs.readFileSync(`${BEST_MODEL_STORAGE}${modelId}.json`, "utf8")));
        model.phenotype = phenotype;
        return model;
    }    
}