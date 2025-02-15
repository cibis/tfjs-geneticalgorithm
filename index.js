const tf = require('@tensorflow/tfjs-node');
var asciichart = require ('asciichart')
const utils = require('./utils')

var geneticAlgorithmConstructor = require("./genetic-algorithm")
var ModelStorage = require("./model-storage/current")
var DataService = require('./data-service');

module.exports = function TFJSGeneticAlgorithmConstructor(options) {
    var startTime, endTime;

    function start() {
        startTime = performance.now();
    };

    function end() {
        endTime = performance.now();
        var timeDiff = endTime - startTime; //in ms 
        // strip the ms 
        timeDiff /= 1000;

        // get seconds 
        var seconds = Math.round(timeDiff);
        var executionTime = new Date(seconds * 1000).toISOString().slice(11, 19);
        console.log(`Execution time ${executionTime}`);
        return executionTime;
    }
    /**
     * The level of mutation. Range 0-1. The bigger the number the more often mutations will happen. Example: _mutationLevel = 0.2 mutations will happen in 2 out of 10 cases
     */
    var _mutationLevel = 0.9;

    function cloneJSON(object) {
        return JSON.parse(JSON.stringify(object))
    }

    function getArrayPercent(array, percent) {
        return array.slice(0, Math.ceil(array.length * percent / 100));
    }

    function crossoverProperties(propA, propB) {
        var r = Math.random();
        if (r < _mutationLevel) return propB;
        return propA;
    }

    function initPhenotype(phenotype){
        if(!phenotype._id) phenotype._id = utils.guidGenerator();
    }

    function settingDefaults() {
        return {
            baseline: undefined,
            modelAbortThreshold: 10,
            /**
             * Training time threshold in seconds
             */
            modelTrainingTimeThreshold: 0,
            parallelProcessing: false,
            parallelism: undefined,
            parameterMutationFunction: undefined,
            modelBuilderFunction: undefined,
            tensors: undefined,
            validationSplit: 0.2,
            populationSize: 100,
            bestPresenceFactor: 2,

            //change it to return the model so it can be saved to the file system and the loss
            modelTrainingFuction: async function (phenotype, model) {
                try {
                    var trainLogs = [];
                    var lossThresholdAbort = false;
                    var errorAbort = false;
                    var trainingStartTime = Date.now();

                    var lossThresholdAbortCnt = 0;

                    const trainDataset =
                        tf.data
                            .generator(
                                () => new DataService.DataSet(
                                    this.tensors.trainingDataSetSource.host,
                                    this.tensors.trainingDataSetSource.path,
                                    this.tensors.trainingDataSetSource.port, 
                                    this.tensors.trainingDataSetSource.cache_id, 
                                    this.tensors.trainingDataSetSource.cache_batch_size, 
                                    { first : (1-this.validationSplit) * 100, batch_size: phenotype.batchSize }
                                ).getNextBatchFunction()
                            )
                    const trainValidationDataset =
                        tf.data
                            .generator(
                                () => new DataService.DataSet(
                                    this.tensors.trainingDataSetSource.host,
                                    this.tensors.trainingDataSetSource.path,
                                    this.tensors.trainingDataSetSource.port, 
                                    this.tensors.trainingDataSetSource.cache_id,  
                                    this.tensors.trainingDataSetSource.cache_batch_size,                                     
                                    { last : this.validationSplit * 100, batch_size: phenotype.batchSize }
                                ).getNextBatchFunction()
                            );                            
                    const valDataset =
                        tf.data
                            .generator(
                                () => new DataService.DataSet(
                                    this.tensors.validationDataSetSource.host,
                                    this.tensors.validationDataSetSource.path,
                                    this.tensors.validationDataSetSource.port,
                                    this.tensors.validationDataSetSource.cache_id,
                                    this.tensors.validationDataSetSource.cache_batch_size,
                                    { batch_size: phenotype.batchSize }                                   
                                ).getNextBatchFunction()
                            );


                    do {
                        var epochTimeStart;
                        await model.fitDataset(trainDataset, {
                            verbose: false,
                            //batchSize: phenotype.batchSize,
                            epochs: phenotype.epochs,
                            //validationSplit: this.validationSplit,
                            validationData: trainValidationDataset,
                            callbacks: {
                                onEpochBegin: (epoch, logs) => {
                                    epochTimeStart = Date.now();
                                },  
                                onEpochEnd: async (epoch, logs) => {
                                    trainLogs.push(logs);
                                    if (isNaN(logs.val_loss)) {
                                        console.log(`Early model loss is NaN abort. Epoch ${epoch} `);
                                        errorAbort = true;
                                        throw Error("IGNORE: Loss is NaN");
                                    }
                                    if (this.modelTrainingTimeThreshold && Date.now() > trainingStartTime + this.modelTrainingTimeThreshold * 1000) {
                                        console.log(`Early model training timeout abort. Epoch ${epoch} `);
                                        errorAbort = true;
                                        throw Error("IGNORE: Model training timeout abort");
                                    }
                                    if (this.modelTrainingTimeThreshold && epochTimeStart && (phenotype.epochs - 1 > epoch)
                                        && ((Date.now() - epochTimeStart) * (phenotype.epochs - epoch - 1)) > this.modelTrainingTimeThreshold * 1000) {
                                        console.log(`Early model training timeout abort based on prior epoch time. Epoch ${epoch} `);
                                        errorAbort = true;
                                        throw Error("Model training timeout abort");
                                      }  
                                    if (this.modelAbortThreshold && trainLogs.length > this.modelAbortThreshold && trainLogs[trainLogs.length - this.modelAbortThreshold].val_loss <= logs.val_loss) {
                                        //console.log(`Early model training abort. Epoch ${epoch}. loss compare ` + trainLogs[trainLogs.length - this.modelAbortThreshold].val_loss + " <= " + logs.val_loss);
                                        lossThresholdAbort = true;
                                        throw Error("IGNORE: Early training abort");
                                    }
                                }
                            }
                        }).catch((err) => {
                            if (!this.lossThresholdAbort && !errorAbort) console.log(err)
                        });
                        if (errorAbort) {
                            return { validationLoss: NaN, phenotype: phenotype };
                        }
                        if (lossThresholdAbort) {
                            phenotype.epoch = trainLogs.length - this.modelAbortThreshold + 1;
                            
                            lossThresholdAbortCnt++;
                            if(lossThresholdAbortCnt > 1){
                              console.log(`Early model training abort`);
                              return { validationLoss: NaN, phenotype: phenotype };
                            }
                            model = settings.modelBuilderFunction(phenotype);
                          }
                    } while (lossThresholdAbort)
                    
                    const result = await model.evaluateDataset(
                        valDataset
                        //, { batchSize: phenotype.batchSize }
                    );
                    const testLoss = parseFloat(result.dataSync()[0].toFixed(4));
                    const trainLoss = trainLogs[trainLogs.length - 1].loss.toFixed(4);
                    const valLoss = trainLogs[trainLogs.length - 1].val_loss.toFixed(4);
                    // console.log("\n============================================================");
                    // model.summary();
                    // console.log(phenotype);
                    //   console.log(
                    //     `Final train-set loss: ${trainLoss}\n` +
                    //     `Final validation-set loss: ${valLoss}\n` +
                    //     `Test-set loss: ${testLoss}`);
                    // console.log("\n============================================================");
                    // var savedModel = await model.save(tf.io.withSaveHandler(async modelArtifacts => modelArtifacts));
                    // savedModel.weightData = Buffer.from(savedModel.weightData).toString("base64");
                    // const modelJson = JSON.stringify(savedModel);
                    phenotype.validationLoss = testLoss;
                    await ModelStorage.writeModel(phenotype._id, model);
                    console.log(`Model training completed ${phenotype._id} . post training eval loss ${testLoss}, training validation-set loss: ${valLoss}, train-set loss: ${trainLoss}`);
                    return { validationLoss: testLoss }
                }
                catch (err) { 
                    console.log(`Error: ${err} stack: ${err.stack}`)
                    return { validationLoss: NaN, phenotype: phenotype };                    
                }
            }
        }
    }

    function settingWithDefaults(settings, defaults) {
        settings = settings || {}

        settings.baseline = settings.baseline || defaults.baseline
        settings.modelAbortThreshold = settings.modelAbortThreshold || defaults.modelAbortThreshold
        settings.modelTrainingTimeThreshold = settings.modelTrainingTimeThreshold || defaults.modelTrainingTimeThreshold
        settings.parallelProcessing = settings.parallelProcessing || defaults.parallelProcessing
        settings.parallelism = settings.parallelism || defaults.parallelism
        settings.validationSplit = settings.validationSplit || defaults.validationSplit
        settings.populationSize = settings.populationSize || defaults.populationSize
        settings.bestPresenceFactor = settings.bestPresenceFactor || defaults.bestPresenceFactor
        
        settings.parameterMutationFunction = settings.parameterMutationFunction || defaults.parameterMutationFunction
        settings.optimizerBuilderFunction = settings.optimizerBuilderFunction || defaults.optimizerBuilderFunction
        settings.modelBuilderFunction = settings.modelBuilderFunction || defaults.modelBuilderFunction
        settings.modelTrainingFuction = settings.modelTrainingFuction || defaults.modelTrainingFuction

        if (!settings.validationSplit) throw Error("validationSplit must be specified. default value 0.2");
        if (!settings.parameterMutationFunction) throw Error("parameterMutationFunction must be defined");
        if (!settings.modelBuilderFunction) throw Error("modelBuilderFunction must be defined");

        return settings
    }

    var settings = settingWithDefaults(options, settingDefaults());

    var ga = geneticAlgorithmConstructor({
        parallelProcessing: settings.parallelProcessing,
        parallelism: settings.parallelism,
        populationSize: settings.populationSize,
        parameterMutationFunction: settings.parameterMutationFunction,
        cloneFunction: async function (phenotype) {
            initPhenotype(phenotype);
            var model = settings.modelBuilderFunction(phenotype);
            var trainingRes = (await settings.modelTrainingFuction(phenotype, model));
            var validationLoss = parseFloat(trainingRes.validationLoss);
            phenotype._type = 'CLONE';
            phenotype.validationLoss = validationLoss;
            //phenotype.modelJson = trainingRes.modelJson;
            return phenotype;
        },        
        mutationFunction: async function (phenotype) {
            phenotype = settings.parameterMutationFunction(phenotype);
            initPhenotype(phenotype);
            var model = settings.modelBuilderFunction(phenotype);
            var validationLoss = (await settings.modelTrainingFuction(phenotype, model)).validationLoss;
            phenotype._id = utils.guidGenerator();
            phenotype._type = 'MUTATION';
            phenotype.validationLoss = parseFloat(validationLoss);
            return phenotype;
        },

        crossoverFunction: async function (a, b) {
            Object.keys(a).forEach(function (akey) {
                Object.keys(b).forEach(function (bkey) {
                    if (akey == bkey) {
                        a[akey] = crossoverProperties(a[akey], b[akey]);
                        b[akey] = crossoverProperties(a[akey], b[akey]);
                    }
                });
            });
            initPhenotype(a);
            initPhenotype(b);
            var amodel = settings.modelBuilderFunction(a);
            var bmodel = settings.modelBuilderFunction(b);
            var aValidationLoss = parseFloat((await settings.modelTrainingFuction(a, amodel)).validationLoss);
            var bValidationLoss = parseFloat((await settings.modelTrainingFuction(b, bmodel)).validationLoss);
            a._id = utils.guidGenerator();
            b._id = utils.guidGenerator();
            a._type = 'CROSSOVER';
            b._type = 'CROSSOVER';
            a.validationLoss = aValidationLoss;
            b.validationLoss = bValidationLoss;
            return [a, b]
        },

        fitnessFunction: function (phenotype) {
            if (typeof phenotype.validationLoss == "undefined" || phenotype.validationLoss == null || isNaN(phenotype.validationLoss)) return NaN;
            return 1 / phenotype.validationLoss;
        },

        doesABeatBFunction: function (a, b) {
            if (isNaN(a.validationLoss)) return b;
            if (isNaN(b.validationLoss)) return a;
            if (a.validationLoss <= b.validationLoss)
                return a;
            else
                return b;
        }
    });

    function filterPopulation(generationsBest){               
        var filteredPopulation =                 
            ga.population()
            .filter(o => !isNaN(o.validationLoss) && (!settings.baseline || o.validationLoss <= settings.baseline));    
            filteredPopulation = filteredPopulation.sort((a,b) => (parseFloat(a.validationLoss) > parseFloat(b.validationLoss)) ? 1 : ((parseFloat(b.validationLoss) > parseFloat(a.validationLoss)) ? -1 : 0));
            filteredPopulation = getArrayPercent(
                filteredPopulation,
                50);                    
            for(var i=0; i < settings.bestPresenceFactor/* generation best next generation presence factor */; i++){
                filteredPopulation = filteredPopulation.concat(generationsBest);
            }            
            filteredPopulation = filteredPopulation.sort((a,b) => (parseFloat(a.validationLoss) > parseFloat(b.validationLoss)) ? 1 : ((parseFloat(b.validationLoss) > parseFloat(a.validationLoss)) ? -1 : 0));
            filteredPopulation = filteredPopulation.slice(0, settings.populationSize);
        // filteredPopulation = getArrayPercent(
        //     filterPopulation,
        //     50);
        console.log(`removing perspectiveless models. removed ${ ga.population().length - filteredPopulation.length }`);
        filteredPopulation = filteredPopulation.length ? filteredPopulation : ga.best();
        ga.updatePopulation(filteredPopulation);
    }

    return {
        ACTIVATIONS: ['elu', 'linear', 'relu', 'relu6', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh'],
        KERNEL_INITIALIZERS: ['leCunNormal', 'glorotNormal', 'glorotUniform', 'heNormal', 'heUniform', 'leCunUniform', 'randomNormal', 'randomUniform', 'truncatedNormal', 'varianceScaling'],
        OPTIMIZERS: ['sgd', 'adagrad', 'adadelta', 'adam', 'adamax', 'rmsprop'],
        
        /**
         * 
         * @param {*} evolveGenerations The number of generations to evolve
         * @param {*} elitesGenerations The number of generations to evolve the best from the of evolution step
         * @returns The model with the smallest loss
         */
        evolve: async function (evolveGenerations, elitesGenerations) {
            _mutationLevel = 0.9;
            start();
            evolveGenerations = evolveGenerations || 1;
            var generationBest = [];
            var plotData = [];
            for (var i = 0; i < evolveGenerations; i++) {
                console.log(`Evolve generation ${i + 1} `)
                console.log("============================================================\n");
                await ga.evolve();
                if(ga.failedPopulation()){
                    console.log(`Evolve generation ${i + 1} FAILED `)
                    continue;
                }
                console.log(`Best in generation ${i + 1} loss: ${ga.best().validationLoss}`)
                plotData.push(parseFloat(ga.best().validationLoss));
                console.log(ga.best());
                generationBest.push(ga.best());
                filterPopulation(generationBest);
                console.log("loss change over generaions");
                console.log(asciichart.plot(plotData));
            }
            if (elitesGenerations) {
                _mutationLevel = 0.5;
                var evolveGenerationBest = cloneJSON(generationBest);                
                generationBest = [];
                for (var i = 0; i < elitesGenerations; i++) {
                    ga.updatePopulation(evolveGenerationBest);
                    console.log(`Finals generation ${i + 1} `)
                    console.log("============================================================\n");
                    await ga.evolve();
                    if(ga.failedPopulation()){
                        console.log(`Evolve generation ${i + 1} FAILED `)
                        continue;
                    }
                    console.log(`Best in finals generation ${i + 1} loss: ${ga.best().validationLoss}`)
                    plotData.push(parseFloat(ga.best().validationLoss));
                    console.log(ga.best());
                    generationBest.push(ga.best());
                    console.log("loss change over generaions");
                    console.log(asciichart.plot(plotData));
                }
            } 

            var minValidationLos = Math.min.apply(Math, generationBest.filter(f => !isNaN(f.validationLoss)).map(m => m.validationLoss));
            var executionTime = end();
            if (isNaN(minValidationLos)) throw Error("All the models failed. Please increase the population size or change the phenotype settings.");
            var bestModel =  generationBest.find(o => o.validationLoss == minValidationLos);
            bestModel.executionTime = executionTime;
            bestModel.evolveGenerations = evolveGenerations;
            bestModel.elitesGenerations = elitesGenerations;
            return bestModel;
        },
        best: function () {
            return ga.best();
        },
        /**
         * Get the clone with the best loss
         * @param {*} phenotype clone parameters
         * @param {*} competitionSize the number of clones to create for the competition
         * @returns 
         */
        cloneCompete: async function(phenotype, competitionSize){
            var modelAbortThreshold = settings.modelAbortThreshold;
            var modelTrainingTimeThreshold = settings.modelTrainingTimeThreshold;
            settings.modelAbortThreshold = null;
            settings.modelTrainingTimeThreshold = null;
            var res = (await ga.cloneCompete(phenotype, competitionSize));
            settings.modelAbortThreshold = modelAbortThreshold;
            settings.modelTrainingTimeThreshold = modelTrainingTimeThreshold;
            return res;
        },

        mutateNumber: function (n, stripDecimals, maximumPercentageChange, alwaysBiggerThanZero, permittedMinimum, permittedMaximum, callStackSize) {
            if (Math.random() > _mutationLevel) return n;
            if (callStackSize && callStackSize > 100) return n;
            var res = n;
            var changeDirection = 1;
            if (Math.random() < 0.5) changeDirection = -1;
            if (res == 0 && Math.random() < 0.5 && !alwaysBiggerThanZero && permittedMinimum == 0) {
                res = maximumPercentageChange / 100 / 2;
            }
            var changeMaximum = res / 100 * maximumPercentageChange;
            var actualChange = Math.random() * 100;
            res = res + (changeMaximum / 100 * actualChange * changeDirection);
            if (stripDecimals) {
                res = Math.round(res);
            }

            if (alwaysBiggerThanZero && res <= 0) res = n;

            if (res < permittedMinimum || res > permittedMaximum) {
                return this.mutateNumber(n, stripDecimals, maximumPercentageChange, alwaysBiggerThanZero, permittedMinimum, permittedMaximum, callStackSize ? callStackSize + 1 : 1);
            }

            return res;
        },

        mutateOptions: function (currentOption, availableOptions) {
            if (Math.random() > _mutationLevel) return currentOption;
            return availableOptions[Math.floor(Math.random() * availableOptions.length)]
        },
        optimizerBuilderFunction: function (optimizer, learningRate) {
            var res;
            switch (optimizer) {
                case "sgd":
                    res = tf.train.sgd(learningRate)
                    break;
                case "adagrad":
                    res = tf.train.adagrad(learningRate)
                    break;
                case "adadelta":
                    res = tf.train.adadelta(learningRate)
                    break;
                case "adam":
                    res = tf.train.adam(learningRate)
                    break;
                case "adamax":
                    res = tf.train.adamax(learningRate)
                    break;
                case "rmsprop":
                    res = tf.train.rmsprop(learningRate)
                    break;
            }
            return res;
        }
    }
}
