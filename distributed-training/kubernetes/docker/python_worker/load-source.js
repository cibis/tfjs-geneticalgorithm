var ExampleDataService = require('../../../../examples/example-data-service');

async function testPredefinedModelsAgainstGA() {
    await ExampleDataService.load();
}

testPredefinedModelsAgainstGA();