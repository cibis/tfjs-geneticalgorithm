module.exports = class ModelStorageInterface {
  constructor() {
    if(!this.readModel) {
      throw new Error("readModel method is not implemented");
    }
    if(!this.writeModel) {
      throw new Error("writeModel method is not implemented");
    }  
    if(!this.listBestModels) {
      throw new Error("writeModel method is not implemented");
    } 
    if(!this.readBestModel) {
      throw new Error("writeModel method is not implemented");
    } 
  }
}