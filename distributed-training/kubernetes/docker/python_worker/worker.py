import os
import uuid
import tensorflow as tf
from pathlib import Path
import urllib.request
import json
import math
from datetime import datetime
from rabbitmq import RabbitMQ
import traceback
from utils import DictToObj
import time


SETTINGS = { "TEST_MODE" : False}


CACHE_STORAGE = "./_runtime/cache/"
MODEL_STORAGE = "shared/models/"

#amqp://rabbitmq-service:5672
RABBITMQ_HOST = None #os.getenv('RABBITMQ_HOST', 'rabbitmq-service')
RABBITMQ_PORT = None #int(os.getenv('RABBITMQ_PORT', 5672))

inputQueue = None #os.environ["JOB_NAME"] + "-INPUT"
outputQueuePrefix = None #os.environ["JOB_NAME"] + "-OUTPUT"

def guidGenerator():
    return str(uuid.uuid4())                                                  

class DataSet:
    def __init__(self, host, path, port, cache_id, cache_batch_size, options):
            self.host = host
            self.path = path
            self.port = port
            self.cache_id = cache_id
            self.cache_batch_size = cache_batch_size
            self.index = 0
            defaultOptions = { 'first': 0, 'last': 0, 'batch_size': 32 }
            self.options = defaultOptions | options
            print(self.options)
            if (int(self.cache_batch_size) <= int(self.options["batch_size"])):
                raise Exception("cache_batch_size should always be bigger than options.batch_size")
            
            self.cacheFileDir = f"{CACHE_STORAGE}{self.cache_id}/"
            self.lastLoadedCacheBatch = None
            self.lastLoadedCacheBatchIndex = None
            self.shape = None
            
    
    def download(self):
         itemsCnt=0
         if(not os.path.isdir(self.cacheFileDir)):
              tmpFolderGuid = guidGenerator()
              self.cacheFileDir = f"{CACHE_STORAGE}{tmpFolderGuid}_{self.cache_id}/"
              path = Path(self.cacheFileDir)
              path.mkdir(parents=True, exist_ok=True)
              batch_index = 0
              source_path = f"{self.path}?index={batch_index}&cache_batch_size={self.cache_batch_size}"
              itemsCnt = 0
              res = { "done": False }
              while(not res["done"]):
                   print(f"http://{self.host}:{self.port}{source_path}")
                   res = json.loads(urllib.request.urlopen(f"http://{self.host}:{self.port}{source_path}").read())
                   with open(f"{self.cacheFileDir}{batch_index}.json", "w") as f:
                        f.write(res["value"])
                   itemsCnt += len(json.loads(res["value"])["xs"])
                   batch_index=batch_index+1
                   source_path = f"{self.path}?index={batch_index}&cache_batch_size={self.cache_batch_size}"
                   if res["done"]:
                        with open(f"{self.cacheFileDir}0.json", "r+") as f:
                            firstBatch = json.loads(f.read())
                            firstBatch["itemsCnt"] = itemsCnt
                            f.seek(0)
                            f.write(json.dumps(firstBatch))
              os.rename(self.cacheFileDir, f"{CACHE_STORAGE}{self.cache_id}/")
              self.cacheFileDir = f"{CACHE_STORAGE}{self.cache_id}/"
         if (itemsCnt==0):
            with open(f"{self.cacheFileDir}0.json", "r") as f:
                self.entierCacheBatch = json.loads(f.read())
                itemsCnt = self.entierCacheBatch["itemsCnt"]
                
         self.maxBatchIndex = math.floor(itemsCnt / self.options["batch_size"]) - 1

         self.minBatchIndex = 0

         if (self.options["first"]):
              if SETTINGS["TEST_MODE"]:     
                self.options["first"] = 0.1

              self.maxBatchIndex = round(self.maxBatchIndex * self.options["first"] / 100)
                
         if (self.options["last"]):
              if SETTINGS["TEST_MODE"]:    
                self.options["last"] = 0.1
              
              self.minBatchIndex = round(self.maxBatchIndex * (100 - self.options["last"]) / 100) + 1    

         print(f"self.minBatchIndex: {self.minBatchIndex}, self.maxBatchIndex: {self.maxBatchIndex}, itemsCnt: {itemsCnt}")    

         self.__next__()
         self.index = 0

    def __iter__(self):
         self.index = 0
         return self

    def __next__(self):
          targetCacheBatchIndex = math.floor(self.index * self.options["batch_size"] / self.cache_batch_size)
          nextCacheBatchIndex = math.floor((self.index + 1) * self.options["batch_size"] / self.cache_batch_size)
          indexWithinTheCacheBatch = ((self.index * self.options["batch_size"]) % self.cache_batch_size)
          moreBatchesInTheCacheBatch = indexWithinTheCacheBatch + self.options["batch_size"] < self.cache_batch_size
          
          self.entierCacheBatch = self.lastLoadedCacheBatch
          if (self.lastLoadedCacheBatchIndex == None or targetCacheBatchIndex != self.lastLoadedCacheBatchIndex):
            with open(f"{self.cacheFileDir}{targetCacheBatchIndex}.json", "r") as f:
                self.entierCacheBatch = json.loads(f.read())
          self.lastLoadedCacheBatch = self.entierCacheBatch
          self.lastLoadedCacheBatchIndex = targetCacheBatchIndex
          itemsLeftInTheCacheBatchToAdd = min(len(self.entierCacheBatch["xs"]) - indexWithinTheCacheBatch, self.options["batch_size"])
          batch = { "xs": self.entierCacheBatch["xs"][slice(indexWithinTheCacheBatch, indexWithinTheCacheBatch + itemsLeftInTheCacheBatchToAdd)], "ys": self.entierCacheBatch["ys"][slice(indexWithinTheCacheBatch, indexWithinTheCacheBatch + itemsLeftInTheCacheBatchToAdd)] }
          file_path = Path(f"{self.cacheFileDir}{nextCacheBatchIndex}.json")
          if (len(batch["xs"]) < self.options["batch_size"] and not moreBatchesInTheCacheBatch and file_path.exists()):
                #since cache batch is bigger than a training batch
                #next cache batch should have enough items for the training batch
                targetCacheBatchIndex=targetCacheBatchIndex+1
                with open(f"{self.cacheFileDir}{targetCacheBatchIndex}.json", "r") as f:
                    entierCacheBatch = json.loads(f.read())
                self.lastLoadedCacheBatch = entierCacheBatch
                self.lastLoadedCacheBatchIndex = targetCacheBatchIndex
                batch["xs"] = batch["xs"] + self.entierCacheBatch["xs"][slice(0, self.options["batch_size"] - len(batch["xs"]))]
                batch["ys"] = batch["ys"] + self.entierCacheBatch["ys"][slice(0, self.options["batch_size"] - len(batch["ys"]))]
          

          self.index=self.index+1
          file_path = Path(f"{self.cacheFileDir}{nextCacheBatchIndex}.json")
          done = self.index > self.maxBatchIndex or (not moreBatchesInTheCacheBatch and not file_path.exists())
          #print(f"self.index: {self.index}, done: {done}, self.maxBatchIndex: {self.maxBatchIndex}, targetCacheBatchIndex: {targetCacheBatchIndex}, nextCacheBatchIndex: {nextCacheBatchIndex}, indexWithinTheCacheBatch: {indexWithinTheCacheBatch}, moreBatchesInTheCacheBatch: {moreBatchesInTheCacheBatch}")
          if self.shape == None:
               vshape = tf.constant(batch["xs"]).shape.as_list()
               vshape[0] = None
               self.input_shape = tf.TensorShape(vshape)
               vshape = tf.constant(batch["ys"]).shape.as_list()
               vshape[0] = None
               self.output_shape = tf.TensorShape(vshape)               
            #    print(f"input shape: {self.input_shape}")
            #    print(f"output shape: {self.output_shape}")
          if done: 
               raise StopIteration
          #return { "value": { "xs": tf.constant(batch["xs"]), "ys": tf.constant(batch["ys"]) }, "done": done }
          return batch["xs"],batch["ys"]


def trainModel(wguid, model, workerdata, trainDS, trainDS_genRes, trainValidationDS, trainValidationDS_genRes, validationDS, validationDS_genRes, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min')  
    history = model.fit(trainDS, epochs=workerdata["phenotype"]["epochs"], steps_per_epoch = trainDS_genRes.maxBatchIndex - trainDS_genRes.minBatchIndex, validation_steps = trainValidationDS_genRes.maxBatchIndex - trainValidationDS_genRes.minBatchIndex,
                        validation_data=trainValidationDS,
                        callbacks=[early_stopping])  
    
    val_performance = model.evaluate(validationDS, return_dict=True)
    loss = val_performance["loss"]
    workerdata["phenotype"]["validationLoss"] = loss
    keras_model_path = f'{CACHE_STORAGE}{wguid}.keras'
    model.save(keras_model_path)
    jsonStr = None
    with open(keras_model_path, 'rb') as file:
        jsonStr = file.read().hex()  

    res = { "validationLoss": loss, "phenotype": workerdata["phenotype"], "modelJson": jsonStr }

    return res

buildModel = None

def readQueue():
    try:
        def callback(ch, method, properties, body):
            global buildModel
            if body != None:
                try:
                    print(body)
                    wguid = guidGenerator()
                    print(f"Worker {wguid} started")
                    message = json.loads(body.decode("utf-8"))
                    workerData = message["workerData"]
                    trainDS_genRes = ds_gen(workerData["tensors"]["trainingDataSetSource"], { "batch_size": workerData["phenotype"]["batchSize"], "first": (1-workerData["validationSplit"]) * 100 })
                    trainValidationDS_genRes = ds_gen(workerData["tensors"]["trainingDataSetSource"], { "batch_size": workerData["phenotype"]["batchSize"], "last": workerData["validationSplit"] * 100 })
                    validationDS_genRes = ds_gen(workerData["tensors"]["validationDataSetSource"], { "batch_size": workerData["phenotype"]["batchSize"] })

                    if buildModel == None:
                        local_ns = {}
                        exec(workerData["modelJson"]["buildModel"], None, local_ns)
                        buildModel = local_ns["buildModel"]
                    model = buildModel(workerData, trainDS_genRes.input_shape)
                    trainDS_genRes = ds_gen(workerData["tensors"]["trainingDataSetSource"], { "batch_size": workerData["phenotype"]["batchSize"], "first": (1-workerData["validationSplit"]) * 100 })
                    trainValidationDS_genRes = ds_gen(workerData["tensors"]["trainingDataSetSource"], { "batch_size": workerData["phenotype"]["batchSize"], "last": workerData["validationSplit"] * 100 })
                    validationDS_genRes = ds_gen(workerData["tensors"]["validationDataSetSource"], { "batch_size": workerData["phenotype"]["batchSize"] })

                    trainDS = tf.data.Dataset.from_generator(generator=lambda:ds_gen(workerData["tensors"]["trainingDataSetSource"], { "batch_size": workerData["phenotype"]["batchSize"], "first": (1-workerData["validationSplit"]) * 100 }),
                        output_types=(tf.float32, tf.float32),
                        output_shapes = (trainDS_genRes.input_shape,trainDS_genRes.output_shape))
                    trainValidationDS = tf.data.Dataset.from_generator(generator=lambda:ds_gen(workerData["tensors"]["trainingDataSetSource"], { "batch_size": workerData["phenotype"]["batchSize"], "last": workerData["validationSplit"] * 100 }),
                        output_types=(tf.float32, tf.float32),
                        output_shapes = (trainValidationDS_genRes.input_shape,trainValidationDS_genRes.output_shape))  
                    validationDS = tf.data.Dataset.from_generator(generator=lambda:ds_gen(workerData["tensors"]["validationDataSetSource"], { "batch_size": workerData["phenotype"]["batchSize"] }),
                        output_types=(tf.float32, tf.float32),
                        output_shapes = (validationDS_genRes.input_shape,validationDS_genRes.output_shape))   
                    trainRes = trainModel(wguid, model, workerData, trainDS, trainDS_genRes, trainValidationDS, trainValidationDS_genRes, validationDS, validationDS_genRes)                 
                    
                    writeQueue(trainRes["phenotype"]["_id"], trainRes)
                except Exception as callback_err:
                    print(traceback.format_exc())
            time.sleep(3)
            #readQueue()

        rabbitmq = RabbitMQ(RABBITMQ_HOST, RABBITMQ_PORT)
        rabbitmq.consume(inputQueue, callback)
        #callback(None, None, None, rabbitmq.get(inputQueue))
    except Exception as err:
        print(traceback.format_exc())
        time.sleep(3)
        print("trying to re-establish connection")
        readQueue()

def writeQueue(id, tfjsJobResponse):
    outputQueue = f"{outputQueuePrefix}-{id}"
    rabbitmq = RabbitMQ(RABBITMQ_HOST, RABBITMQ_PORT)
    rabbitmq.publish(outputQueue, json.dumps(tfjsJobResponse))
    rabbitmq.close()  

def ds_gen(data_source, options):
     trainDS = DataSet(data_source["host"], data_source["path"], data_source["port"], data_source["pre_cache"], data_source["cache_batch_size"], options)
     trainDS.download()
     return trainDS

def optimizerBuilderFunction(optimizer, learningRate):
    print(f'optimizerBuilderFunction({optimizer}, {learningRate})')
    match optimizer:
         case "sgd":
              return tf.keras.optimizers.SGD(learning_rate=learningRate)
         case "adagrad":
              return tf.keras.optimizers.Adagrad(learning_rate=learningRate)
         case "adadelta":
              return tf.keras.optimizers.Adadelta(learning_rate=learningRate)
         case "adam":
              return tf.keras.optimizers.Adam(learning_rate=learningRate)
         case "adamax":
              return tf.keras.optimizers.Adamax(learning_rate=learningRate)
         case "rmsprop":
              return tf.keras.optimizers.RMSprop(learning_rate=learningRate)  

def initializers(initializer):
    match initializer:
        case 'leCunNormal':
            return tf.keras.initializers.LecunNormal()
        case 'glorotNormal':
            return tf.keras.initializers.GlorotNormal()
        case 'glorotUniform':
            return tf.keras.initializers.GlorotUniform()
        case 'heNormal':
            return tf.keras.initializers.HeNormal()
        case 'heUniform':
            return tf.keras.initializers.HeUniform()
        case 'leCunUniform':
            return tf.keras.initializers.LecunUniform()
        case 'randomNormal':
            return tf.keras.initializers.RandomNormal()       
        case 'randomUniform':
            return tf.keras.initializers.RandomUniform()    
        case 'truncatedNormal':
            return tf.keras.initializers.TruncatedNormal()    
        case 'varianceScaling':
            return tf.keras.initializers.VarianceScaling()                                                                    

def lossBuilderFunction(loss):
     match loss:
          case "meanAbsoluteError":
               return tf.keras.losses.MeanAbsoluteError()
          case "meanSquaredError":
               return tf.keras.losses.MeanSquaredError()

def main():  
    global SETTINGS, RABBITMQ_HOST, RABBITMQ_PORT, inputQueue, outputQueuePrefix
    print(f"main() -> SETTINGS: {SETTINGS}")

    if SETTINGS["TEST_MODE"]:
        RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', '127.0.0.1')
        RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 30000))        
    else:
        RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'rabbitmq-service')
        RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5672))
    inputQueue = os.environ["JOB_NAME"] + "-INPUT"
    outputQueuePrefix = os.environ["JOB_NAME"] + "-OUTPUT"
    print(f"inputQueue: {inputQueue}")
    print(f"outputQueuePrefix: {outputQueuePrefix}")
    readQueue()