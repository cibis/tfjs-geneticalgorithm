def buildModel(workerData):
    #tf = importlib.import_module("tensorflow")
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])    

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.RMSprop(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])   
    return model 