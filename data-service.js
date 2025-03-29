const http = require('http')
const tf = require('@tensorflow/tfjs-node');
const utils = require('./utils');
const fs = require('fs');
var os = require("os");


const CACHE_STORAGE = "./_runtime/cache/"

class DataSetSources {
    constructor(trainingDataSetSource, validationDataSetSource) {
        this.trainingDataSetSource = trainingDataSetSource;
        this.validationDataSetSource = validationDataSetSource;
    }
}

class DataSetSource {
    constructor(host, path, port, pre_cache, cache_batch_size) {
        this.host = host;
        this.path = path;
        this.port = port;
        this.pre_cache = pre_cache;
        if (this.pre_cache) {
            if (typeof pre_cache == "boolean")
                this.cache_id = utils.guidGenerator();
            else
                this.cache_id = pre_cache;
            this.cache_batch_size = cache_batch_size;
        }
    }
}

class DataSet {
    /**
     * 
     * @param {*} host 
     * @param {*} path 
     * @param {*} port 
     * @param {*} options { first: val, last: val }; first - Take only first portion of the data set. Acceptable value range 0 - 100; last - Take only last portion of the data set. Acceptable value range 0 - 100
     */
    constructor(host, path, port, cache_id, cache_batch_size, options) {
        this.host = host;
        this.path = path;
        this.port = port;
        this.cache_id = cache_id;
        this.cache_batch_size = cache_batch_size;
        this.index = 0;
        var defaultOptions = { first: 0, last: 0, batch_size: 32 };
        this.options = Object.assign(defaultOptions, options);   
        if(parseInt(this.cache_batch_size) <= parseInt(this.options.batch_size))    
            throw Error("cache_batch_size should always be bigger than options.batch_size");  
    }

    async getNextBatchFunction() {
        var itemsCnt = 0;
        var lastLoadedCacheBatch = null;
        var lastLoadedCacheBatchIndex = null;
        if (this.cache_id) {
            var cacheFileDir = `${CACHE_STORAGE}${this.cache_id}/`;
            if (!fs.existsSync(cacheFileDir)) {
                var tmpFolderGuid = utils.guidGenerator();
                cacheFileDir = `${CACHE_STORAGE}${tmpFolderGuid + "_"}${this.cache_id}/`;
                fs.mkdirSync(cacheFileDir, { recursive: true });
                var batch_index = 0;
                var asyncRequest = () => {
                    var options = {
                        host: this.host,
                        port: this.port,
                        path: `${this.path}${this.path.indexOf("?") != -1 ? "&" : "?"}index=${batch_index}&cache_batch_size=${this.cache_batch_size}`,
                        method: 'GET'
                    };
                    return new Promise(function (resolve, reject) {
                        var req = http.get(options, function (res) {
                            var bodyChunks = [];
                            res.on('data', function (chunk) {
                                bodyChunks.push(chunk);
                            }).on('end', function () {
                                var body = Buffer.concat(bodyChunks);
                                resolve(JSON.parse(body));
                            })
                        });

                        req.on('error', function (e) {
                            console.log('ERROR: ' + e.message);
                            reject(e);
                        });
                    });
                }
                var res;
                itemsCnt = 0;
                do {
                    var res = await asyncRequest(batch_index);
                    fs.writeFileSync(`${cacheFileDir}${batch_index}.json`, res.value);
                    itemsCnt += JSON.parse(res.value).xs.length;
                    batch_index++;
                    if(res.done) {
                        var firstBatch = JSON.parse(fs.readFileSync(`${cacheFileDir}0.json`));
                        firstBatch.itemsCnt = itemsCnt;
                        fs.writeFileSync(`${cacheFileDir}0.json`, JSON.stringify(firstBatch))
                    }
                } while (!res.done);
                console.log('completed downloading. renaming folder');
                fs.renameSync(cacheFileDir, `${CACHE_STORAGE}${this.cache_id}/`);
                console.log('completed renaming folder');
                cacheFileDir = `${CACHE_STORAGE}${this.cache_id}/`;
            }

            this.options.first = parseFloat(this.options.first);
            this.options.last = parseFloat(this.options.last);
            
            if (!itemsCnt){
                var entierCacheBatch = JSON.parse(fs.readFileSync(`${cacheFileDir}0.json`));
                itemsCnt = entierCacheBatch.itemsCnt;
            }
            var maxBatchIndex = Math.floor(itemsCnt / this.options.batch_size)-1;
            
            var minBatchIndex = 0;
            
            if(this.options.first){
                maxBatchIndex = Math.round(maxBatchIndex * this.options.first / 100);
            }  
            if(this.options.last){
                minBatchIndex = Math.round(maxBatchIndex * (100 - this.options.last) / 100) + 1;
            } 
            
            this.index = minBatchIndex;
            const iterator = {
                
                next: async () => {
                    var targetCacheBatchIndex = Math.floor(this.index * this.options.batch_size / this.cache_batch_size);
                    var nextCacheBatchIndex = Math.floor((this.index + 1) * this.options.batch_size / this.cache_batch_size);
                    var indexWithinTheCacheBatch = ((this.index * this.options.batch_size) % this.cache_batch_size);
                    var moreBatchesInTheCacheBatch = indexWithinTheCacheBatch + this.options.batch_size < this.cache_batch_size;                       
                    var entierCacheBatch = lastLoadedCacheBatch;
                    if (lastLoadedCacheBatchIndex == null || targetCacheBatchIndex != lastLoadedCacheBatchIndex)
                        entierCacheBatch = JSON.parse(fs.readFileSync(`${cacheFileDir}${targetCacheBatchIndex}.json`));
                    lastLoadedCacheBatch = entierCacheBatch;
                    lastLoadedCacheBatchIndex = targetCacheBatchIndex;
                    var itemsLeftInTheCacheBatchToAdd = Math.min(entierCacheBatch.xs.length - indexWithinTheCacheBatch, this.options.batch_size);
                    var batch = { xs: entierCacheBatch.xs.slice(indexWithinTheCacheBatch, indexWithinTheCacheBatch + itemsLeftInTheCacheBatchToAdd), ys: entierCacheBatch.ys.slice(indexWithinTheCacheBatch, indexWithinTheCacheBatch + itemsLeftInTheCacheBatchToAdd) };
                    if(batch.xs.length < this.options.batch_size && !moreBatchesInTheCacheBatch && fs.existsSync(`${cacheFileDir}${nextCacheBatchIndex}.json`)){
                        //since cache batch is bigger than a training batch
                        //next cache batch should have enough items for the training batch
                        targetCacheBatchIndex++;                        
                        entierCacheBatch = JSON.parse(fs.readFileSync(`${cacheFileDir}${targetCacheBatchIndex}.json`));
                        lastLoadedCacheBatch = entierCacheBatch;
                        lastLoadedCacheBatchIndex = targetCacheBatchIndex;
                        batch.xs = batch.xs.concat(entierCacheBatch.xs.slice(0, this.options.batch_size - batch.xs.length));
                        batch.ys = batch.ys.concat(entierCacheBatch.ys.slice(0, this.options.batch_size - batch.ys.length));
                    }
                    
                    this.index++;
                    var done = this.index > maxBatchIndex || (!moreBatchesInTheCacheBatch && !fs.existsSync(`${cacheFileDir}${nextCacheBatchIndex}.json`));
                    return { value: { xs: tf.tensor(batch.xs), ys: tf.tensor(batch.ys) }, done: done };                    
                }
            };

            return iterator;

        }
        else {
            const iterator = {
                next: async () => {
                    var options = {
                        host: this.host,
                        port: this.port,
                        path: `${this.path}${this.path.indexOf("?") != -1 ? "&" : "?"}index=${this.index}&first=${this.options.first}&last=${this.options.last}&batch_size=${this.options.batch_size}`,
                        method: 'GET'
                    };
                    var asyncRequest = () => {
                        return new Promise(function (resolve, reject) {
                            var req = http.get(options, function (res) {
                                var bodyChunks = [];
                                res.on('data', function (chunk) {
                                    bodyChunks.push(chunk);
                                }).on('end', function () {
                                    var body = Buffer.concat(bodyChunks);
                                    resolve(JSON.parse(body));
                                })
                            });

                            req.on('error', function (e) {
                                console.log('ERROR: ' + e.message);
                                reject(e);
                            });
                        });
                    }
                    var res = await asyncRequest(this.index);
                    this.index++;

                    return { value: { xs: tf.tensor(res.value.xs), ys: tf.tensor(res.value.ys) }, done: res.done };
                }
            };
            return iterator;
        }
    }

}

module.exports = {
    DataSet,
    DataSetSource,
    DataSetSources
}