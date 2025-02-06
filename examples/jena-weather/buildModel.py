#the imports are not required here. use 
# tf = importlib.import_module("tensorflow")
#inside the method instead
import tensorflow as tf
from utils import DictToObj

def buildModel(workerData, inputShape):
    phenotype = DictToObj(workerData["phenotype"])

    model = tf.keras.Sequential()

    if phenotype.modelType == "linear":
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=1))
    elif phenotype.modelType == "mlp":
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=phenotype.hiddenLayerUnits, activation=phenotype.activation, kernel_initializer=initializers(phenotype.kernelInitializer)))

        for i in range(1, phenotype.hiddenLayers):
            model.add(tf.keras.layers.Dense(units=phenotype.hiddenLayerUnits, activation=phenotype.activation, kernel_initializer=initializers(phenotype.kernelInitializer)))
        
        if phenotype.add_dropout and phenotype.dropoutRate > 0:
            model.add(tf.keras.layers.Dropout(rate=phenotype.dropoutRate))
    elif phenotype.modelType == "simpleRNN":
        model.add(tf.keras.layers.SimpleRNN(units=phenotype.hiddenLayerUnits, activation=phenotype.activation, kernel_initializer=initializers(phenotype.kernelInitializer), dropout=(phenotype.dropoutRate if phenotype.add_dropout else 0), recurrent_dropout=(phenotype.recurrentDropout if phenotype.add_dropout else 0)))
    elif phenotype.modelType == "gru":
        model.add(tf.keras.layers.GRU(units=phenotype.hiddenLayerUnits, activation=phenotype.activation, kernel_initializer=initializers(phenotype.kernelInitializer), dropout=(phenotype.dropoutRate if phenotype.add_dropout else 0), recurrent_dropout=(phenotype.recurrentDropout if phenotype.add_dropout else 0)))
    elif phenotype.modelType == "lstm":
        model.add(tf.keras.layers.LSTM(units=phenotype.hiddenLayerUnits, activation=phenotype.activation, kernel_initializer=initializers(phenotype.kernelInitializer), dropout=(phenotype.dropoutRate if phenotype.add_dropout else 0), recurrent_dropout=(phenotype.recurrentDropout if phenotype.add_dropout else 0)))

    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer=optimizerBuilderFunction(phenotype.optimizer, phenotype.learningRate), loss=lossBuilderFunction(phenotype.loss), metrics=[lossBuilderFunction(phenotype.loss)])

    return model