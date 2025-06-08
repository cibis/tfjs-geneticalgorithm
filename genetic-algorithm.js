const utils = require('./utils')

module.exports = function geneticAlgorithmConstructor(options) {

    function settingDefaults() {
        return {
            parallelProcessing: false,
            parallelism: undefined,

            cloneFunction: undefined,

            mutationFunction: async function (phenotype) { return phenotype },

            crossoverFunction: async function (a, b) { return [a, b] },

            fitnessFunction: function (phenotype) { return 0 },

            doesABeatBFunction: undefined,

            population: [],
            populationSize: 100,
        }
    }

    function settingWithDefaults(settings, defaults) {
        settings = settings || {}

        settings.parallelProcessing = settings.parallelProcessing || defaults.parallelProcessing
        settings.parallelism = settings.parallelism || defaults.parallelism
        settings.cloneFunction = settings.cloneFunction || defaults.cloneFunction
        settings.mutationFunction = settings.mutationFunction || defaults.mutationFunction
        settings.crossoverFunction = settings.crossoverFunction || defaults.crossoverFunction
        settings.fitnessFunction = settings.fitnessFunction || defaults.fitnessFunction

        settings.doesABeatBFunction = settings.doesABeatBFunction || defaults.doesABeatBFunction

        settings.population = settings.population || defaults.population
        // if ( settings.population.length <= 0 ) throw Error("population must be an array and contain at least 1 phenotypes")

        settings.populationSize = settings.populationSize || defaults.populationSize
        if (settings.populationSize <= 0) throw Error("populationSize must be greater than 0")

        return settings
    }

    var settings = settingWithDefaults(options, settingDefaults())

    function waitForCondition(conditionFn, interval = 1000) {
        return new Promise(resolve => {
            const checkCondition = () => {
                if (conditionFn()) {
                    resolve();
                } else {
                    setTimeout(checkCondition, interval);
                }
            };
            checkCondition();
        });
    }

    function removeArrayItem(array, item) {
        var index = array.indexOf(item);
        if (index !== -1) {
            array.splice(index, 1);
        }
    }

    async function populate() {
        var size = settings.population.length
        var addNewPhenotype = async () => {
            var newItem = await mutate(
                settings.population.length ?
                    cloneJSON(settings.population[Math.floor(Math.random() * size)]) :
                    null
            );
            newItem._type = 'MUTATION';
            newItem._id = utils.guidGenerator();

            settings.population.push(
                newItem
            )
        }

        var getPhenotypeToClone = () => {
            if(!settings.population.length){
                return settings.parameterMutationFunction();
            }
            return settings.population[Math.floor(Math.random() * size)];
        }

        if (settings.parallelProcessing) {
            var requiredPopulation = settings.populationSize - settings.population.length;
            var promises = [];
            for (var i = 0; i < requiredPopulation; i++) {
                var newPromise = mutate(cloneJSON(getPhenotypeToClone()));
                promises.push(newPromise);
                newPromise.then(newItem=>{
                        newItem._type = 'MUTATION';
                        newItem._id = utils.guidGenerator();
    
                        settings.population.push(
                            newItem
                        )
                    removeArrayItem(promises, newPromise); 
                });                
                if (settings.parallelism && settings.parallelism > 0/* && (promises.length == settings.parallelism || (promises.length > 0 && i == requiredPopulation - 1))*/) {
                    // var responses = await Promise.all(promises);
                    // responses.forEach(newItem => {
                    //     newItem._type = 'MUTATION';
                    //     newItem._id = utils.guidGenerator();
    
                    //     settings.population.push(
                    //         newItem
                    //     )
                    // });
                    // promises = [];
                    await waitForCondition(()=> promises.length < settings.parallelism)
                }       
            }
            if (promises.length) {
                await Promise.all(promises);
                // var responses = await Promise.all(promises);
                // responses.forEach(newItem => {
                //     newItem._type = 'MUTATION';
                //     newItem._id = utils.guidGenerator();

                //     settings.population.push(
                //         newItem
                //     )
                // });
            }
        }
        else {
            while (settings.population.length < settings.populationSize) {
                await addNewPhenotype();
            }
        }
        //console.log(settings.population);
    }

    function cloneJSON(object) {
        return JSON.parse(JSON.stringify(object))
    }

    async function mutate(phenotype) {
        phenotype = cloneJSON(phenotype);
        if(phenotype) phenotype._id = utils.guidGenerator();
        return await settings.mutationFunction(phenotype)
    }

    async function crossover(phenotype) {
        phenotype = cloneJSON(phenotype)
        var mate = settings.population[Math.floor(Math.random() * settings.population.length)]
        mate = cloneJSON(mate)
        mate._id = utils.guidGenerator();
        return (await settings.crossoverFunction(phenotype, mate))[0]
    }

    function doesABeatB(a, b) {
        var doesABeatB = false;
        if (settings.doesABeatBFunction) {
            return settings.doesABeatBFunction(a, b)
        } else {
            return settings.fitnessFunction(a) >= settings.fitnessFunction(b)
        }
    }

    async function compete() {
        var nextGeneration = []
        var promises = [];

        for (var p = 0; p < settings.population.length - 1; p += 2) {
            var phenotype = settings.population[p];
            var competitor = settings.population[p + 1];

            nextGeneration.push(phenotype)

            if (doesABeatB(phenotype, competitor)) {
                var newPromise;
                if (Math.random() < 0.5) {
                    if (settings.parallelProcessing){
                        newPromise = mutate(phenotype);
                        promises.push(newPromise);    
                    }               
                    else
                        nextGeneration.push(await mutate(phenotype));
                } 
                else {
                    if (settings.parallelProcessing){
                        newPromise = crossover(phenotype);
                        promises.push(newPromise);
                    }
                    else {
                        var cp = await crossover(phenotype);
                        nextGeneration.push(cp);
                    }
                }
                newPromise.then(r=>{
                    nextGeneration.push(r);
                    removeArrayItem(promises, newPromise);                
                })
                if (settings.parallelProcessing && settings.parallelism && settings.parallelism > 0 /*&& ((promises.length == settings.parallelism * 3) || (promises.length > 0 && p == settings.population.length - 2))*/) {
                    // var responses = await Promise.all(promises);
                    // nextGeneration.push(...responses);
                    // promises = [];
                    await waitForCondition(()=> promises.length < settings.parallelism)
                }     
            } else {
                nextGeneration.push(competitor)
            }
        }
        if (promises.length) {
            await Promise.all(promises);
            // var responses = await Promise.all(promises);
            // nextGeneration.push(...responses);
        }

        settings.population = nextGeneration;
    }

    function randomizePopulationOrder() {

        for (var index = 0; index < settings.population.length; index++) {
            var otherIndex = Math.floor(Math.random() * settings.population.length)
            var temp = settings.population[otherIndex]
            settings.population[otherIndex] = settings.population[index]
            settings.population[index] = temp
        }
    }

    return {
        evolve: async function (options) {
            if (options) {
                settings = settingWithDefaults(options, settings)
            }

            await populate()
            randomizePopulationOrder()
            await compete()
            return this
        },
        failedPopulation: function () {
            return !this.scoredPopulation().length;
        },        
        best: function () {
            var scored = this.scoredPopulation()
            var result = scored.reduce(function (a, b) {
                return a.score >= b.score ? a : b
            }, scored[0]).phenotype
            return cloneJSON(result)
        },
        cloneCompete: async function (phenotype, competitionSize) {
            try {                
                if (!settings.cloneFunction) throw Error("cloneFunction must be defined to use cloneCompete method");
                var res = [];
                var promises = [];
                for (var i = 0; i < competitionSize; i++) {
                    phenotype = cloneJSON(phenotype);
                    phenotype._id = utils.guidGenerator();

                    if (settings.parallelProcessing) {  
                        var newPromise = settings.cloneFunction(phenotype);
                        promises.push(newPromise);    
                        newPromise.then(r=>{
                            res.push(r);
                            removeArrayItem(promises, newPromise);
                        });                  
                        if (settings.parallelism && settings.parallelism > 0/* && (promises.length == settings.parallelism || (promises.length > 0 && i == competitionSize - 1))*/) {
                            // var responses = await Promise.all(promises);
                            // res.push(...responses);
                            // promises = [];
                            await waitForCondition(()=> promises.length < settings.parallelism)
                        }
                    }                    
                    else {
                        res.push(await settings.cloneFunction(phenotype));
                    }
                }
                if (promises.length) {
                    await Promise.all(promises);
                    //var responses = await Promise.all(promises);
                    //res.push(...responses);
                }
                console.log("all promises completed");

                var scored = res.map(function (phenotype) {
                    return {
                        phenotype: cloneJSON(phenotype),
                        score: settings.fitnessFunction(phenotype)
                    }
                }).filter(o => !isNaN(o.score));
                if (!scored.length) throw Error("All the trained models failed");
                var result = scored.reduce(function (a, b) {
                    return a.score >= b.score ? a : b
                }, scored[0]).phenotype;
            }
            catch (err) {
                console.log(`Error in cloneCompete ${err.message}  stack ${err.stack}`);
                throw err;
            }
            return result;
        },
        bestScore: function () {
            return settings.fitnessFunction(this.best())
        },
        population: function () {
            return cloneJSON(this.config().population)
        },
        updatePopulation: function (newPopulation) {
            settings.population = cloneJSON(newPopulation);
        },
        scoredPopulation: function () {
            return this.population().map(function (phenotype) {
                return {
                    phenotype: cloneJSON(phenotype),
                    score: settings.fitnessFunction(phenotype)
                }
            }).filter(o => !isNaN(o.score))
        },
        config: function () {
            return cloneJSON(settings)
        },
        clone: function (options) {
            return geneticAlgorithmConstructor(
                settingWithDefaults(options,
                    settingWithDefaults(this.config(), settings)
                )
            )
        }
    }
}
