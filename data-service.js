const http = require('http')
const tf = require('@tensorflow/tfjs-node');

class DataSetSources {
    constructor(trainingDataSetSource, validationDataSetSource) {
        this.trainingDataSetSource = trainingDataSetSource;
        this.validationDataSetSource = validationDataSetSource;
    }
}

class DataSetSource {
    constructor(host, path, port) {
        this.host = host;
        this.path = path;
        this.port = port;
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
    constructor(host, path, port, options) {
        this.host = host;
        this.path = path;
        this.port = port;
        this.index = 0;
        this.options = options ?? {first : 0, last : 0};
        
        if(typeof this.options.first == 'undefined' || isNaN(this.options.first)) this.options.first = 0;
        this.options.first = parseFloat(this.options.first);
        
        if(typeof this.options.last == 'undefined' || isNaN(this.options.last)) this.options.last = 0; 
        this.options.last = parseFloat(this.options.last);        
    }

    getNextBatchFunction() {
        const iterator = {
            next: async () => {
                var options = {
                    host: this.host,
                    port: this.port,
                    path: `${this.path}?index=${this.index}&first=${this.options.first}&last=${this.options.last}`,
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

module.exports = {
    DataSet,
    DataSetSource,
    DataSetSources
}