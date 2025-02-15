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

    async function populate() {
        var size = settings.population.length
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
                promises.push(mutate(cloneJSON(getPhenotypeToClone())));
                if (settings.parallelism && settings.parallelism > 0 && (promises.length == settings.parallelism || (promises.length > 0 && i == requiredPopulation - 1))) {
                    var responses = await Promise.all(promises);
                    responses.forEach(newItem => {
                        newItem._type = 'MUTATION';
                        newItem._id = utils.guidGenerator();
    
                        settings.population.push(
                            newItem
                        )
                    });
                    promises = [];
                }       
            }
            if (promises.length) {
                var responses = await Promise.all(promises);
                responses.forEach(newItem => {
                    newItem._type = 'MUTATION';
                    newItem._id = utils.guidGenerator();

                    settings.population.push(
                        newItem
                    )
                });
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
                if (Math.random() < 0.5) {
                    if (settings.parallelProcessing)
                        promises.push(mutate(phenotype));                   
                    else
                        nextGeneration.push(await mutate(phenotype));
                } 
                else {
                    if (settings.parallelProcessing)
                        promises.push(crossover(phenotype));
                    else {
                        var cp = await crossover(phenotype);
                        nextGeneration.push(cp);
                    }
                }
                if (settings.parallelProcessing && settings.parallelism && settings.parallelism > 0 && (promises.length == settings.parallelism || (promises.length > 0 && p == settings.population.length - 2))) {
                    var responses = await Promise.all(promises);
                    nextGeneration.push(...responses);
                    promises = [];
                }     
            } else {
                nextGeneration.push(competitor)
            }
        }
        if (promises.length) {
            var responses = await Promise.all(promises);
            nextGeneration.push(...responses);
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
                        promises.push(settings.cloneFunction(phenotype));                      
                        if (settings.parallelism && settings.parallelism > 0 && (promises.length == settings.parallelism || (promises.length > 0 && i == competitionSize - 1))) {
                            var responses = await Promise.all(promises);
                            res.push(...responses);
                            promises = [];
                        }
                    }                    
                    else {
                        res.push(await settings.cloneFunction(phenotype));
                    }
                }
                if (promises.length) {
                    var responses = await Promise.all(promises);
                    res.push(...responses);
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
