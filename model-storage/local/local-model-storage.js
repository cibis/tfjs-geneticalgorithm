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

    async writeModel(modelId, model){
        if(!modelId) throw Error("modelId is undefined");
        if (!fs.existsSync(BEST_MODEL_STORAGE)){
            fs.mkdirSync(BEST_MODEL_STORAGE, { recursive: true });
        }
        if (!fs.existsSync(MODEL_STORAGE)){
            fs.mkdirSync(MODEL_STORAGE, { recursive: true });
        }
        //console.log(`${Date.now()} LocalModelStorage writeModel ${modelId} ${!!model}`);
        await model.save(`file://${MODEL_STORAGE}${modelId}`);
    }

    copyToBest(phenotype, group){
        phenotype.group = group;
        fs.writeFile(`${BEST_MODEL_STORAGE}${phenotype._id}.json`, JSON.stringify(phenotype), 'utf8',()=>{}); 
        fs.cp(MODEL_STORAGE + phenotype._id, BEST_MODEL_STORAGE + phenotype._id, {recursive: true}, (err) => {/* callback */});
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