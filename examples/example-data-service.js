const tf = require('@tensorflow/tfjs-node');
const express = require("express");
const app = express();
const BostonHousing = require('./boston-housing/boston-housing')
var bostonDataTensors = BostonHousing.getTensor();
var bostonData = BostonHousing.getBostonData();

function getArrayPart(arr, first, last, length){
   
    if(typeof first == 'undefined' || isNaN(first)) first = 0;
    first = parseFloat(first);
    
    if(typeof last == 'undefined' || isNaN(last)) last = 0; 
    last = parseFloat(last);

    if(first){
        var indexAtPercentage = Math.round((length - 1) * first / 100);
        return arr.slice([0], [indexAtPercentage]);
    }  
    if(last){
        var indexAtPercentage = Math.round((length - 1) * (100 - last) / 100);
        return arr.slice([indexAtPercentage]);
    }     
    return arr;
}

app.get("/boston-housing-training", (req, res) => {
    res.send({ value: { 
        xs: getArrayPart(bostonDataTensors.trainFeatures, req.query.first, req.query.last, bostonData.trainFeatures.length).arraySync(), 
        ys: getArrayPart(bostonDataTensors.trainTarget, req.query.first, req.query.last, bostonData.trainFeatures.length).arraySync() 
    }, done: true });
});

app.get("/boston-housing-validation", (req, res) => {
    res.send({ value: { 
        xs: getArrayPart(bostonDataTensors.testFeatures, req.query.first, req.query.last, bostonData.testFeatures.length).arraySync(), 
        ys: getArrayPart(bostonDataTensors.testTarget, req.query.first, req.query.last, bostonData.testFeatures.length).arraySync() 
    }, done: true });
});

app.listen(3000, () => console.log("Server is listening on port 3000"));
