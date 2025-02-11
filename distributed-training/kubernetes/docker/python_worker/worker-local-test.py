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

#SETTINGS["TEST_MODE"] = True
os.environ["RABBITMQ_HOST"] = "127.0.0.1"
os.environ["RABBITMQ_PORT"] = "30000"    
os.environ["JOB_NAME"] = "tfjsjob"
main()