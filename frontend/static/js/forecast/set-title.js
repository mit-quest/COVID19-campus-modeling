var queryDict = {};
window.location.search.substr(1).split("&").forEach(function(item) {
    queryDict[item.split("=")[0]] = item.split("=")[1]
});

if (queryDict.hasOwnProperty("uuid")) {
    globalTitle = 'Forecast - ' + queryDict['uuid'];
} else {
    globalTitle = 'Select Forecast';
}

document.title = globalTitle;
