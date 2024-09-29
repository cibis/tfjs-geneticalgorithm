
Train/Evolve/Optimize tensorflow js models with a genetic algorithm. Forked from [panchishin/geneticalgorithm](https://github.com/panchishin/geneticalgorithm)

Original code was updated to enable asyncrounous and distributed evolution with extra functionality to facilitate tensorflow.js model structure and hyperparameter evolution and optimization.

Genetic Algorithms(GA) when using for model evolution is a time consuming process so one of the main problems was to speed up the process by using asyncrounus and distributed model traing.  This package covers three execution modes:
- Single Thread
- Node.js Workers
- Kubernetes Job(minikube)

GA evolution resulting models are stored in the _runtime\best subfolder and can be tested using 
```
node examples\boston-housing\run-best-saved-models.js
```

The examples/boston-housing sub-folder allows you to run the [boston-housing](https://github.com/tensorflow/tfjs-examples/blob/master/boston-housing) example against the GA alghoritm 
- Single Thread: 
```
node examples\boston-housing\run-basic.js
```
- Node.js Workers:
```
node examples\boston-housing\run-workers.js
```
- Kubernetes Job(minikube. requires some [pre-configuration](#kubernetes):
```
node examples\boston-housing\run-kubernetes.js
```
![](https://github.com/cibis/tfjs-geneticalgorithm/blob/master/_runtime/screenshots/loss-chart.jpg?raw=true)
![](https://github.com/cibis/tfjs-geneticalgorithm/blob/master/_runtime/screenshots/final.jpg?raw=true)

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
        tensors: BostonHousing.getTensor(),
        parameterMutationFunction: (oldPhenotype) => {
			//build new phenotype. check examples\boston-housing\run-*** for examples
			return newPhenotype;
		},
		modelBuilderFunction: (phenotype) => { 
			//build tfjs model based on phenotype parameters. check examples\boston-housing\run-*** for examples
			return model;
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
# Coming-Next
- Adding the more complicated jena-weather example that includes mlp,mlp-l2,linear-regression, mlp-dropout, simpleRNN and gru model types
- Using files for passing large traning data to the workers
- Adding the possibility for custom training data loading functionality on the worker