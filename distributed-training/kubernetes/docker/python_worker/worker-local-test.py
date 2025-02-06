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
from worker import main
from worker import SETTINGS

SETTINGS["TEST_MODE"] = True
os.environ["RABBITMQ_HOST"] = "127.0.0.1"
os.environ["RABBITMQ_PORT"] = "30000"    
os.environ["JOB_NAME"] = "tfjsjob"
# RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'rabbitmq-service')
# RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5672))
# with open("buildModel.py", 'r') as file:
#     buildModel = file.read()      
#     workerData =    {
#                     "workerData": { 
#                         "modelJson":{
#                                         "buildModel": buildModel
#                                     },
#                         "phenotype": {
#                                         "modelType": "mlp",
#                                         "add_dropout": False,
#                                         "dropoutRate": 0,
#                                         "recurrentDropout": 0,
#                                         "epochs": 1,
#                                         "batchSize": 128,
#                                         "learningRate": 0.01,
#                                         "hiddenLayers": 1,
#                                         "hiddenLayerUnits": 32,
#                                         "activation": "relu",
#                                         "kernelInitializer": "leCunNormal",
#                                         "optimizer": "rmsprop",
#                                         "loss": "meanAbsoluteError",
#                                         "_id": "0cfb3118-640e-e65b-7b00-3f40d779f51e",
#                                         "validationLoss": 14.2844,
#                                         "_type": "CLONE",
#                                         "executionTime": "00:43:32",
#                                         "evolveGenerations": 5,
#                                         "elitesGenerations": 2,
#                                         "group": "boston-housing"
#                                     }, 
#                         "tensors":  {
#                                         "trainingDataSetSource": {
#                                             "host": "127.0.0.1",
#                                             "path": "/jena-weather-training",
#                                             "port": "3000",
#                                             "pre_cache": "jena-weather-training",
#                                             "cache_id": "jena-weather-training",
#                                             "cache_batch_size": 1280
#                                         },
#                                         "validationDataSetSource": {
#                                             "host": "127.0.0.1",
#                                             "path": "/jena-weather-validation",
#                                             "port": "3000",
#                                             "pre_cache": "jena-weather-validation",
#                                             "cache_id": "jena-weather-validation",
#                                             "cache_batch_size": 1280
#                                         }
#                                     },
#                         "validationSplit": 0.2, 
#                         "modelAbortThreshold": None,
#                         "modelTrainingTimeThreshold": None
#                     }
#                 }
# rabbitmq = RabbitMQ(RABBITMQ_HOST, RABBITMQ_PORT)
# rabbitmq.publish("tfjsjob-INPUT", json.dumps(workerData))
# rabbitmq.close() 
# print(os.environ["JOB_NAME"])
main()