const tf = require('@tensorflow/tfjs-node');
const express = require("express");
const app = express();
const BostonHousing = require('./boston-housing/boston-housing')
var bostonDataTensors = BostonHousing.getTensor();
var bostonData = BostonHousing.getBostonData();
const JenaWeather = require('./jena-weather/jena-weather')
var jenaWeatherData = new JenaWeather.JenaWeatherData();


function getArrayPart(arr, batch_size, index, length){
    batch_size = parseInt(batch_size);
    index = parseInt(index);    
    var adjustedLength = Math.floor(arr.arraySync().length / batch_size) * batch_size;
    arr = arr.slice([0], [adjustedLength]);
    arr = tf.split(arr, adjustedLength / batch_size)    

    return arr[index];
}

app.get("/boston-housing-training", (req, res) => {
    const cache_batch_size = parseInt(req.query.cache_batch_size);
    var batchIndex = parseInt(req.query.index);
    console.log(`get boston-housing-training data: ${cache_batch_size * batchIndex} out of ${bostonData.trainFeatures.length}`);
    res.send({ value: JSON.stringify({ 
        xs: getArrayPart(bostonDataTensors.trainFeatures, cache_batch_size, batchIndex, bostonData.trainFeatures.length).arraySync(), 
        ys: getArrayPart(bostonDataTensors.trainTarget, cache_batch_size, batchIndex, bostonData.trainFeatures.length).arraySync() 
    }), done: !getArrayPart(bostonDataTensors.trainFeatures, cache_batch_size, batchIndex + 1, bostonData.trainFeatures.length) });
});

app.get("/boston-housing-validation", (req, res) => {
    const cache_batch_size = parseInt(req.query.cache_batch_size);
    var batchIndex = parseInt(req.query.index);
    console.log(`get boston-housing-validation data: ${cache_batch_size * batchIndex} out of ${bostonData.testFeatures.length}`);
    res.send({ value: JSON.stringify({ 
        xs: getArrayPart(bostonDataTensors.testFeatures, cache_batch_size, batchIndex, bostonData.testFeatures.length).arraySync(), 
        ys: getArrayPart(bostonDataTensors.testTarget, cache_batch_size, batchIndex, bostonData.testFeatures.length).arraySync() 
    }), done: !getArrayPart(bostonDataTensors.testFeatures, cache_batch_size, batchIndex + 1, bostonData.testFeatures.length) });
});


// app.get("/jena-weather-training-OLD", (req, res) => {    
//     var TRAIN_MIN_ROW = 0;
//     var TRAIN_MAX_ROW = 250000;
//     const trainShuffle = true;
//     const lookBack = 10 * 24 * 6;  // Look back 10 days.
//     const step = 6;                // 1-hour steps.
//     const delay = 24 * 6;          // Predict the weather 1 day later.
//     const batchSize = 1280;
//     const normalize = true;
//     const includeDateTime = false;
    
//     var first = req.query.first;
//     var last = req.query.last;
//     if(typeof first == 'undefined' || isNaN(first)) first = 0;
//     first = parseFloat(first);
    
//     if(typeof last == 'undefined' || isNaN(last)) last = 0; 
//     last = parseFloat(last);

//     if(first && last) throw new Error("first and last cannot be set both at the same time");
//     if(first){
//         TRAIN_MAX_ROW = Math.round((250000 - 1) * first / 100);
//     }  
//     if(last){
//         TRAIN_MIN_ROW = Math.round((250000 - 1) * (100 - last) / 100);
//     }     
    
//     var targetIndex = parseInt(req.query.index);
//     console.log(`/jena-weather-training START targetIndex: ${targetIndex}, TRAIN_MAX_ROW: ${TRAIN_MAX_ROW}, TRAIN_MIN_ROW: ${TRAIN_MIN_ROW}, Current: ${TRAIN_MIN_ROW + (batchSize * targetIndex)}`);

//     var generator = jenaWeatherData.getNextBatchFunction(
//         trainShuffle, lookBack, delay, batchSize, step, TRAIN_MIN_ROW + (batchSize * targetIndex),
//         TRAIN_MAX_ROW, normalize, includeDateTime);


//     var currentBatch;
//     //for (var i = 0; i <= targetIndex; i++) {
//         currentBatch = generator.next();
//     //}
    

//     res.send({ value: { 
//         xs: currentBatch.value.xs.arraySync(), 
//         ys: currentBatch.value.ys.arraySync() 
//     }, done: ((TRAIN_MIN_ROW + (batchSize * targetIndex + 1)) >  TRAIN_MAX_ROW) });
//     console.log("/jena-weather-training END");
// });

app.get("/jena-weather-training", (req, res) => {    
    var TRAIN_MIN_ROW = 0;
    var TRAIN_MAX_ROW = 250000;
    const trainShuffle = true;
    const lookBack = 10 * 24 * 6;  // Look back 10 days.
    const step = 6;                // 1-hour steps.
    const delay = 24 * 6;          // Predict the weather 1 day later.
    const cache_batch_size = parseInt(req.query.cache_batch_size);
    const normalize = true;
    const includeDateTime = false;
    
   
    var batchIndex = parseInt(req.query.index);

    console.log(`get jena-weather-training data: ${TRAIN_MIN_ROW + (cache_batch_size * batchIndex)} out of ${TRAIN_MAX_ROW}`);

    var generator = jenaWeatherData.getNextBatchFunction(
        trainShuffle, lookBack, delay, cache_batch_size, step, TRAIN_MIN_ROW + (cache_batch_size * batchIndex),
        TRAIN_MAX_ROW, normalize, includeDateTime);

    var currentBatch = generator.next(); 
    res.send({ value: JSON.stringify({
        xs: currentBatch.value.xs.arraySync(), 
        ys: currentBatch.value.ys.arraySync() 
    }), done: ((TRAIN_MIN_ROW + (cache_batch_size * (batchIndex + 1))) >  TRAIN_MAX_ROW) });
});

app.get("/jena-weather-validation", (req, res) => {
    const VAL_MIN_ROW = 250001;
    const VAL_MAX_ROW = 300000;
    const evalShuffle = false;
    const lookBack = 10 * 24 * 6;  // Look back 10 days.
    const step = 6;                // 1-hour steps.
    const delay = 24 * 6;          // Predict the weather 1 day later.
    const cache_batch_size = parseInt(req.query.cache_batch_size);
    const normalize = true;
    const includeDateTime = false;

    var batchIndex = parseInt(req.query.index);

    console.log(`get jena-weather-validation data: ${VAL_MIN_ROW + (cache_batch_size * batchIndex)} out of ${VAL_MAX_ROW}`);

    var generator = jenaWeatherData.getNextBatchFunction(
        evalShuffle, lookBack, delay, cache_batch_size, step, VAL_MIN_ROW + (cache_batch_size * batchIndex),
        VAL_MAX_ROW, normalize, includeDateTime);


    var currentBatch = generator.next();

    res.send({
        value: JSON.stringify({
            xs: currentBatch.value.xs.arraySync(),
            ys: currentBatch.value.ys.arraySync()
        }), 
        done: ((VAL_MIN_ROW + (cache_batch_size * (batchIndex + 1))) >  VAL_MAX_ROW)
    });
});

async function loadDataAndRun(params) {
    return new Promise(function (resolve, reject) {
        bostonData.loadData().then(() => {
            BostonHousing.arraysToTensors();
            jenaWeatherData.load().then(() => {
                app.listen(3000, () => console.log("Server is listening on port 3000"));
                setTimeout(() => {
                    resolve();
                }, 1000);
            });
        })
    });
}

module.exports.load =  loadDataAndRun;


