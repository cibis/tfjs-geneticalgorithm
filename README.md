
Train/Evolve/Optimize tensorflow js models with a genetic algorithm. Forked from [panchishin/geneticalgorithm](https://github.com/panchishin/geneticalgorithm)

Original code was updated to enable asynchronous and distributed evolution with extra functionality to facilitate tensorflow.js model structure and hyperparameter evolution and optimization.

Genetic Algorithms(GA) when using for model evolution is a time consuming process so one of the main problems was to speed up the process by using asyncrounus and distributed model training.  This package covers three execution modes:
- Single Thread
- Node.js Workers
- Kubernetes Job(minikube)

GA evolution resulting models are stored in the _runtime\best subfolder and can be tested using 
```
node examples\boston-housing\run-best-saved-models.js
```

The examples/boston-housing sub-folder allows you to run the [boston-housing](https://github.com/tensorflow/tfjs-examples/blob/master/boston-housing) example against the GA algorithm 
- Single Thread: 
```
node examples\boston-housing\run-basic.js
```
- Node.js Workers:
```
node examples\boston-housing\run-workers.js
```
- Kubernetes Job(minikube. requires some [pre-configuration](#kubernetes)):
```
node examples\boston-housing\run-kubernetes.js
```
![](_runtime/screenshots/loss-chart.jpg?raw=true)
![](_runtime/screenshots/final.jpg?raw=true)

Section Links : [Constructor](#constructor) , [Kubernetes](#kubernetes), [Tests](#tests), [Coming-Next](#coming-next)
# Constructor
### TFJSGeneticAlgorithmConstructor constructor
```js
var TFJSGeneticAlgorithmConstructor = require('tfjs-geneticalgorithm')
var ga = TFJSGeneticAlgorithmConstructor( config )

```
The minimal configuration for constructing a GeneticAlgorithm calculator is like so:

```js
var config = {
        populationSize: taskSettings.populationSize,
        baseline: taskSettings.baseline,
        tensors: new DataService.DataSetSources(
            new DataService.DataSetSource("127.0.0.1", "/boston-housing-training", "3000", "boston-housing-training", 333),
            new DataService.DataSetSource("127.0.0.1", "/boston-housing-validation", "3000", "boston-housing-validation", 173)
        ),
        parameterMutationFunction: (oldPhenotype) => {
          //build new phenotype. check examples\boston-housing\run-*** for examples
          return newPhenotype;
        },
        modelBuilderFunction: (phenotype) => { 
          //build tfjs model based on phenotype parameters. check examples\boston-housing\run-*** for examples
          return model;
        },

        /* below settings required only for distributed processing */
        parallelProcessing: true,
        /* number of parallel workers */
        parallelism: taskSettings.parallelism,
        modelTrainingFuction: async () =>{
          /*
            worker object from:
            distributed-training/kubernetes/worker_starter/WorkerTraining - for kubernetes
            distributed-training/workers/worker_starter/WorkerTraining - for nodejs workers
          */
          var workerResponse = await worker.trainModel(phenotype, modelJson, this.tensors, this.validationSplit, this.modelAbortThreshold, this.modelTrainingTimeThreshold);
          phenotype.epochs = workerResponse.phenotype.epochs;
          return { validationLoss: workerResponse.validationLoss }
        }
}

var ga = TFJSGeneticAlgorithmConstructor( config )

//evolve for 5 generations and return the best model
var bestModel = await ga.evolve(5);

//optional. create 100 clones of the same model. train them and get the one with the best results/weights
bestModel = await ga.cloneCompete(bestModel, 100);
```
# Tests
```
npm test
```

# Kubernetes
The kubernetes example was tested locally using minikube and docker. In order to run the example the following commands need to be executed
```
kubectl create -f https://kubernetes.io/examples/application/job/rabbitmq/rabbitmq-service.yaml
kubectl create -f https://kubernetes.io/examples/application/job/rabbitmq/rabbitmq-statefulset.yaml
kubectl port-forward service/rabbitmq-service 30000:5672

cd distributed-training\kubernetes\docker\worker
docker build -t tfjs-ks-worker .

minikube image load tfjs-ks-worker:latest

cd distributed-training\kubernetes
kubectl apply -f pv-claim.yml
kubectl apply -f pv-volume.yml
```
### How it works
WorkerTraining class is creating a job in the kubernetes cluster and the job in its turn is creating a number of pods matching the parallelism parameter. The communication with the workers running in the pods is done using rabbitmq running in the cluster. After the evolution operation is completed the job is deleted together with the pods.
![](_runtime/screenshots/kb.jpg?raw=true)

# Jan 19 2025
Switched model input data to DataSet
Added Jena Weather example. To run use any of these(very slow)
```
node examples\jena-weather\run-basic.js
node examples\jena-weather\run-workers.js
```

# Coming-Next
- Kubernetes only - support for tensorflow python workers