# cd distributed-training\kubernetes\docker\python_worker
# docker build -t tfjs-ks-python-worker .
# minikube image load tfjs-ks-python-worker:latest
#try: minikube image build -t tfjs-ks-python-worker -f ./Dockerfile .

FROM tensorflow/tensorflow
COPY . .
RUN pip install pika
#RUN pip install seaborn
#RUN pip install -r requirements.txt
CMD [ "python", "worker-start.py"]