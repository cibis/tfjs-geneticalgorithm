module.exports = class DistributedTrainingInterface {
  constructor() {
    if(!this.trainModel) {
      throw new Error("trainModel method is not implemented");
    }
  }
}