# cd distributed-training\kubernetes\docker\python_worker
# docker build -t tfjs-ks-python-worker .
# minikube image load tfjs-ks-python-worker:latest

FROM tensorflow/tensorflow
COPY . .
RUN pip install pandas
RUN pip install seaborn
#RUN pip install -r requirements.txt
CMD [ "python", "worker.py"]