function addSectionBreak(questionID, title, description) {

  questionDIV = document.getElementById('view-' + questionID);

  contentDIV = document.createElement('div');
  contentDIV.setAttribute("class", "col-12");

  hr = document.createElement('hr');
  contentDIV.appendChild(hr);

  titleElement = document.createElement('h2');
  titleElement.innerHTML = title;
  contentDIV.appendChild(titleElement);

  descriptionElement = document.createElement('p');
  descriptionElement.innerHTML = description;
  contentDIV.appendChild(descriptionElement);

  questionDIV.appendChild(contentDIV);

}

function initializeDataViz(size, questionID, plotName, linkedPlotNames, linkedControls) {
  console.log("begin initializeDataViz");
  console.log({ "initializeDataViz": 1, "size": size, "questionID": questionID, "plotName": plotName, "linkedPlotNames": linkedPlotNames, "linkedControls": linkedControls });

  // create the elements

  if (true) {

    if (plotName == 'campus-rampup' || plotName == 'community-symptoms' || plotName == 'test-results' || plotName == 'test-lag' || plotName == 'test-staleness') {
      plotTitle = linkedControls;
    }

    if (plotName == 'image' || plotName == 'list') {
      plotID = linkedPlotNames;
      plotTitle = linkedControls;
      graphicsID = questionID + '-' + plotName + '-' + plotID;
    } else {
      graphicsID = questionID + '-' + plotName;
    }

    data = plotData[graphicsID];
    console.log({'key': 'initializeDataViz', 'graphicsID': graphicsID, 'data': data});

    questionDIV = document.getElementById('view-' + questionID);

    bufferDIV = document.createElement('div');
    bufferDIV.classList.add(size);
    bufferDIV.classList.add('graphic-buffer');

    contentDIV = document.createElement('div');
    contentDIV.classList.add('graphic-pane');

    bufferDIV.appendChild(contentDIV);
    questionDIV.appendChild(bufferDIV);

    graphicsDIV = document.createElement('div');
    graphicsDIV.classList.add('graphics-container');
    graphicsContentsDIV = document.createElement('div');
    graphicsContentsDIV.id = graphicsID;
    graphicsDIV.appendChild(graphicsContentsDIV);

  }

  // SIMULATION-1: DEMOGRAPHICS

  if (plotName == 'total-population-size') {

    // title and controls
    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Total population size';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    // graphics
    contentDIV.appendChild(graphicsDIV);
    updateDataViz(questionID, plotName, 'n/a');

  }

  if (plotName == 'age-bin-chart') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Number of people in each age bin';
    controlDIV.appendChild(textElement);
    contentDIV.appendChild(controlDIV);
    contentDIV.appendChild(graphicsDIV);

    updateDataViz(questionID, plotName, 'n/a');
  }

  if (plotName == 'job-category-chart') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Number of people in each job category';
    controlDIV.appendChild(textElement);
    contentDIV.appendChild(controlDIV);
    contentDIV.appendChild(graphicsDIV);

    updateDataViz(questionID, plotName, 'n/a');
  }

  if (plotName == 'home-zipcode-chart') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Number of people in each zip code';
    controlDIV.appendChild(textElement);
    contentDIV.appendChild(controlDIV);
    contentDIV.appendChild(graphicsDIV);

    updateDataViz(questionID, plotName, 'n/a');
  }

  if (plotName == 'commute-type-chart') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Number of people using each commute type';
    controlDIV.appendChild(textElement);
    contentDIV.appendChild(controlDIV);
    contentDIV.appendChild(graphicsDIV);

    updateDataViz(questionID, plotName, 'n/a');
  }

  // SIMULATION-2 BUILDING USAGE
  if (plotName == 'campus-visits-time-plot') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Distinct individual visits of people to campus over time';
    controlDIV.appendChild(textElement);
    contentDIV.appendChild(controlDIV);
    contentDIV.appendChild(graphicsDIV);

    updateDataViz(questionID, plotName, 'n/a');

  }

  if (plotName == 'campus-inflow-time-plot') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Inflow of people onto campus over time';
    controlDIV.appendChild(textElement);
    contentDIV.appendChild(controlDIV);
    contentDIV.appendChild(graphicsDIV);

    updateDataViz(questionID, plotName, 'n/a');

  }

  if (plotName == 'campus-outflow-time-plot') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Outflow of people from campus over time';
    controlDIV.appendChild(textElement);
    contentDIV.appendChild(controlDIV);
    contentDIV.appendChild(graphicsDIV);

    updateDataViz(questionID, plotName, 'n/a');

  }

  if (plotName == 'building-visits-time-plot') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Distinct individual visits of people in building' + '&nbsp;';
    controlDIV.appendChild(textElement);

    controlElement = document.createElement('select');
    controlElement.classList.add("custom-select");
    controlElement.id = 'control-' + questionID + '-' + plotName;
    controlDIV.appendChild(controlElement);

    textElement = document.createElement('span');
    textElement.innerHTML = '&nbsp;' + 'over time';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    values = Object.keys(data[Object.keys(data)[0]]);
    console.log({ "values": values });
    values.forEach(value => {
      element = document.createElement('option');
      element.value = value;
      element.innerHTML = value;
      controlElement.appendChild(element);
    });

    controlElement.value = values[0];
    controlElement.onchange = function(){
      linkedPlotNames.forEach(linkedPlotName => {
        updateDataViz(questionID, linkedPlotName, this.value);
      });
      linkedControls.forEach(linkedControl => {
        document.getElementById(linkedControl).value = this.value;
      });
    };
    contentDIV.appendChild(graphicsDIV);

  }

  if (plotName == 'building-density-time-plot') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Density of people in building' + '&nbsp;';
    controlDIV.appendChild(textElement);

    controlElement = document.createElement('select');
    controlElement.setAttribute("class", "custom-select");
    controlElement.id = 'control-' + questionID + '-' + plotName;
    controlDIV.appendChild(controlElement);

    textElement = document.createElement('span');
    textElement.innerHTML = '&nbsp;' + 'over time';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    values = Object.keys(data[Object.keys(data)[0]]);
    console.log({ "values": values });
    values.forEach(value => {
      element = document.createElement('option');
      element.value = value;
      element.innerHTML = value;
      controlElement.appendChild(element);
    });

    controlElement.value = values[0];
    controlElement.onchange = function(){
      linkedPlotNames.forEach(linkedPlotName => {
        updateDataViz(questionID, linkedPlotName, this.value);
      });
      linkedControls.forEach(linkedControl => {
        document.getElementById(linkedControl).value = this.value;
      });
    };
    contentDIV.appendChild(graphicsDIV);

  }

  if (plotName == 'building-inflow-time-plot') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Inflow of people into building' + '&nbsp;';
    controlDIV.appendChild(textElement);

    controlElement = document.createElement('select');
    controlElement.setAttribute("class", "custom-select");
    controlElement.id = 'control-' + questionID + '-' + plotName;
    controlDIV.appendChild(controlElement);

    textElement = document.createElement('span');
    textElement.innerHTML = '&nbsp;' + 'over time';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    values = Object.keys(data[Object.keys(data)[0]]);
    console.log({ "values": values });
    values.forEach(value => {
      element = document.createElement('option');
      element.value = value;
      element.innerHTML = value;
      controlElement.appendChild(element);
    });

    controlElement.value = values[0];
    controlElement.onchange = function(){
      linkedPlotNames.forEach(linkedPlotName => {
        updateDataViz(questionID, linkedPlotName, this.value);
      });
      linkedControls.forEach(linkedControl => {
        document.getElementById(linkedControl).value = this.value;
      });
    };
    contentDIV.appendChild(graphicsDIV);

  }

  if (plotName == 'building-outflow-time-plot') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Outflow of people from building' + '&nbsp;';
    controlDIV.appendChild(textElement);

    controlElement = document.createElement('select');
    controlElement.setAttribute("class", "custom-select");
    controlElement.id = 'control-' + questionID + '-' + plotName;
    controlDIV.appendChild(controlElement);

    textElement = document.createElement('span');
    textElement.innerHTML = '&nbsp;' + 'over time';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    values = Object.keys(data[Object.keys(data)[0]]);
    console.log({ "values": values });
    values.forEach(value => {
      element = document.createElement('option');
      element.value = value;
      element.innerHTML = value;
      controlElement.appendChild(element);
    });

    controlElement.value = values[0];
    controlElement.onchange = function(){
      linkedPlotNames.forEach(linkedPlotName => {
        updateDataViz(questionID, linkedPlotName, this.value);
      });
      linkedControls.forEach(linkedControl => {
        document.getElementById(linkedControl).value = this.value;
      });
    };
    contentDIV.appendChild(graphicsDIV);

  }

  // SIMULATION-3: HARMS

  if (plotName == 'all-people-chart') {

    // title and controls
    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Harms for all people';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    // graphics
    contentDIV.appendChild(graphicsDIV);
    updateDataViz(questionID, plotName, 'n/a');

  }

  if (plotName == 'harms-time-plot') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Daily count of people experiencing' + '&nbsp;';
    controlDIV.appendChild(textElement);

    controlElement = document.createElement('select');
    controlElement.setAttribute("class", "custom-select");
    controlElement.id = 'control-' + questionID + '-' + plotName;
    controlDIV.appendChild(controlElement);

    textElement = document.createElement('span');
    textElement.innerHTML = '';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    values = Object.keys(data[Object.keys(data)[0]]);
    console.log({ "values": values });
    values.forEach(value => {
      element = document.createElement('option');
      element.value = value;
      element.innerHTML = value;
      controlElement.appendChild(element);
    });

    controlElement.value = values[0];
    controlElement.onchange = function(){
      linkedPlotNames.forEach(linkedPlotName => {
        updateDataViz(questionID, linkedPlotName, this.value);
      });
      linkedControls.forEach(linkedControl => {
        document.getElementById(linkedControl).value = this.value;
      });
    };
    contentDIV.appendChild(graphicsDIV);

  }

  if (plotName == 'age-group-table') {

    // title and controls
    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Harms split by age group';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    // graphics
    contentDIV.appendChild(graphicsDIV);
    updateDataViz(questionID, plotName, 'n/a');
  }

  if (plotName == 'age-group-chart-row') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Harms to people between' + '&nbsp;';
    controlDIV.appendChild(textElement);

    controlElement = document.createElement('select');
    controlElement.setAttribute("class", "custom-select");
    controlElement.id = 'control-' + questionID + '-' + plotName;
    controlDIV.appendChild(controlElement);

    textElement = document.createElement('span');
    textElement.innerHTML = '&nbsp;' + 'years old';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    values = Object.keys(data);
    values.forEach(value => {
      element = document.createElement('option');
      element.value = value;
      element.innerHTML = value;
      controlElement.appendChild(element);
    });

    controlElement.value = values[0];
    controlElement.onchange = function(){
      linkedPlotNames.forEach(linkedPlotName => {
        updateDataViz(questionID, linkedPlotName, this.value);
      });
      linkedControls.forEach(linkedControl => {
        document.getElementById(linkedControl).value = this.value;
      });
    };
    contentDIV.appendChild(graphicsDIV);

  }

  if (plotName == 'age-group-chart-column') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Total number of people in each age group who will experience' + '&nbsp;';
    controlDIV.appendChild(textElement);

    controlElement = document.createElement('select');
    controlElement.setAttribute("class", "custom-select");
    controlElement.id = 'control-' + questionID + '-' + plotName;
    controlDIV.appendChild(controlElement);

    textElement = document.createElement('span');
    textElement.innerHTML = '';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    values = Object.keys(data[Object.keys(data)[0]]);
    values.forEach(value => {
      element = document.createElement('option');
      element.value = value;
      element.innerHTML = value;
      controlElement.appendChild(element);
    });

    controlElement.value = values[0];
    controlElement.onchange = function(){
      linkedPlotNames.forEach(linkedPlotName => {
        updateDataViz(questionID, linkedPlotName, this.value);
      });
      linkedControls.forEach(linkedControl => {
        document.getElementById(linkedControl).value = this.value;
      });
    };
    contentDIV.appendChild(graphicsDIV);

  }

  if (plotName == 'job-category-table') {

    // title and controls
    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Harms split by job category';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    // graphics
    contentDIV.appendChild(graphicsDIV);
    updateDataViz(questionID, plotName, 'n/a');
  }

  if (plotName == 'job-category-chart-row') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Harms experienced by' + '&nbsp;';
    controlDIV.appendChild(textElement);

    controlElement = document.createElement('select');
    controlElement.setAttribute("class", "custom-select");
    controlElement.id = 'control-' + questionID + '-' + plotName;
    controlDIV.appendChild(controlElement);

    textElement = document.createElement('span');
    textElement.innerHTML = '';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    values = Object.keys(data);
    values.forEach(value => {
      element = document.createElement('option');
      element.value = value;
      element.innerHTML = value;
      controlElement.appendChild(element);
    });

    controlElement.value = values[0];
    controlElement.onchange = function(){
      linkedPlotNames.forEach(linkedPlotName => {
        updateDataViz(questionID, linkedPlotName, this.value);
      });
      linkedControls.forEach(linkedControl => {
        document.getElementById(linkedControl).value = this.value;
      });
    };
    contentDIV.appendChild(graphicsDIV);

  }

  if (plotName == 'job-category-chart-column') {

    controlDIV = document.createElement('form');
    controlDIV.setAttribute("class", "d-flex justify-content-center text-center form-inline plot-title");

    textElement = document.createElement('span');
    textElement.innerHTML = 'Total number of people in each job category who will experience' + '&nbsp;';
    controlDIV.appendChild(textElement);

    controlElement = document.createElement('select');
    controlElement.setAttribute("class", "custom-select");
    controlElement.id = 'control-' + questionID + '-' + plotName;
    controlDIV.appendChild(controlElement);

    textElement = document.createElement('span');
    textElement.innerHTML = '';
    controlDIV.appendChild(textElement);

    contentDIV.appendChild(controlDIV);

    values = Object.keys(data[Object.keys(data)[0]]);
    values.forEach(value => {
      element = document.createElement('option');
      element.value = value;
      element.innerHTML = value;
      controlElement.appendChild(element);
    });

    controlElement.value = values[0];
    controlElement.onchange = function(){
      linkedPlotNames.forEach(linkedPlotName => {
        updateDataViz(questionID, linkedPlotName, this.value);
      });
      linkedControls.forEach(linkedControl => {
        document.getElementById(linkedControl).value = this.value;
      });
    };
    contentDIV.appendChild(graphicsDIV);

  }

  console.log("end initializeDataViz");

}

function updateDataViz(questionID, plotName, selection) {
    console.log("begin updateDataViz");
    console.log({ "questionID": questionID, "plotName": plotName, "selection": selection });

    elemID = questionID + '-' + plotName; //plotID already added to plotName above for image and list

    // SIMULATION-1: DEMOGRAPHICS

    if (plotName == 'total-population-size') {
      displayText(elemID);
    };

    if (plotName == 'age-bin-chart') {
      ageBinChart(elemID);
    };

    if (plotName == 'job-category-chart') {
      jobCategoryChart(elemID);
    };

    if (plotName == 'home-zipcode-chart') {
      homeZipcodeChart(elemID);
    };

    if (plotName == 'commute-type-chart') {
      commuteTypeChart(elemID);
    };

    // SIMULATION-2: BUILDING USAGE

    if (plotName == 'campus-visits-time-plot') {
      campusTimePlot('Distinct Individual Visits (Persons)', elemID);
    };

    if (plotName == 'campus-inflow-time-plot') {
      campusTimePlot('Inflow (Persons/Hour)', elemID)
    };

    if (plotName == 'campus-outflow-time-plot') {
      campusTimePlot('Outflow (Persons/Hour)', elemID)
    };

    if (plotName == 'building-visits-time-plot') {
      buildingTimePlot('Distinct Individual Visits (Persons)', selection, elemID);
    };

    if (plotName == 'building-density-time-plot') {
      buildingTimePlot('Density (Persons/Sq. Ft of Usable Building Area)', selection, elemID);
    };

    if (plotName == 'building-inflow-time-plot') {
      buildingTimePlot('Inflow (Persons/Hour)', selection, elemID)
    };

    if (plotName == 'building-outflow-time-plot') {
      buildingTimePlot('Outflow (Persons/Hour)', selection, elemID)
    };

    // SIMULATION-3: HARMS

    if (plotName == 'all-people-chart') {
      allPeopleChart(elemID);
    };

    if (plotName == 'harms-time-plot') {
      harmsTimePlot(selection, elemID)
    };

    if (plotName == 'age-group-table') {
      ageGroupTable(elemID);
    };

    if (plotName == 'age-group-chart-row') {
      ageGroupChartRow(selection, elemID);
    };

    if (plotName == 'age-group-chart-column') {
      ageGroupChartColumn(selection, elemID);
    };

    if (plotName == 'job-category-table') {
      jobCategoryTable(elemID);
    }

    if (plotName == 'job-category-chart-row') {
      jobCategoryChartRow(selection, elemID);
    };

    if (plotName == 'job-category-chart-column') {
      jobCategoryChartColumn(selection, elemID);
    };

    // kludgy fix to enforce proper graphic scaling
    window.dispatchEvent(new Event('resize'));

    console.log("end updateDataViz");
}

// VISUALIZATION FUNCTIONS

function displayt0(t0Date, nDays) {
  elem = document.getElementById('t0-li');
  elem.setAttribute('class', 'navbar-item');
  elem = document.getElementById('t0-a');
  elem.setAttribute('class', 'navbar-link btn btn-text disabled text-right');
  elem.innerHTML = 'Simulation starting on ' + t0Date + ' for ' + nDays + ' days<br/>' + globalUUID;
}

// SIMULATION-1: DEMOGRAPHICS
function displayText(elemID) {
  console.log('begin displayText');
  console.log({ 'elemID': elemID, 'data': data  });
  elem = document.getElementById(elemID);
  elem.classList.add('text-center');
  elem.classList.add('display-text');
  elem.innerHTML = data;
  console.log('end displayText');
}

function ageBinChart(elemID) {
  console.log("begin ageBinChart");
  console.log({ 'elemID': elemID });

  data = plotData[elemID];

  xValues = Object.keys(data);
  console.log({ 'xValues': xValues });

  base = Object.values(data);
  console.log({ 'base': base });

  tenPercentile = [];
  for (key in base) {
    tenPercentile.push(base[key]['10 percentile']);
  }

  fiftyPercentile = [];
  for (key in base) {
    fiftyPercentile.push(base[key]['50 percentile']);
  }

  nintyPercentile = [];
  for (key in base) {
    nintyPercentile.push(base[key]['90 percentile']);
  }


  deltaAbove = [];
  fiftyPercentile.forEach((element, index) => {
    deltaAbove[index] = nintyPercentile[index] - fiftyPercentile[index];
  });

  deltaBelow = [];
  fiftyPercentile.forEach((element, index) => {
    deltaBelow[index] = fiftyPercentile[index] - tenPercentile[index];
  });

  chart = {
    x: xValues,
    y: fiftyPercentile,
    name: 'Control',
    error_y: {
      type: 'data',
      symmetric: false,
      array: deltaAbove,
      arrayminus: deltaBelow
    },
    type: 'bar'
  };

  charts = [chart];

  layout = {
    margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }, //hide title area
    dragmode: false,
    xaxis: {
      title: 'Age Bin',
      automargin: true,
      titlefont: {
        family: 'Arial, sans-serif',
        size: 18,
        color: 'lightgrey'
      }
    },
    yaxis: {
      title: 'Number of People',
      automargin: true,
      titlefont: {
        family: 'Arial, sans-serif',
        size: 18,
        color: 'lightgrey'
      }
    },
    autosize: true,
    width: '90%'
  };

  config = {
    responsive: true,
    displayModeBar:false,
    scrollZoom: false
  }

  Plotly.newPlot(elemID, charts, layout, config);

  console.log("end ageBinChart");
}

function jobCategoryChart(elemID) {
  console.log("begin ageBinChart");
  console.log({ 'elemID': elemID });

  data = plotData[elemID];

  xValues = Object.keys(data);

  base = Object.values(data);
  console.log({ 'base': base });

  tenPercentile = [];
  for (key in base) {
    tenPercentile.push(base[key]['10 percentile']);
  }

  fiftyPercentile = [];
  for (key in base) {
    fiftyPercentile.push(base[key]['50 percentile']);
  }

  nintyPercentile = [];
  for (key in base) {
    nintyPercentile.push(base[key]['90 percentile']);
  }


  deltaAbove = [];
  fiftyPercentile.forEach((element, index) => {
    deltaAbove[index] = nintyPercentile[index] - fiftyPercentile[index];
  });

  deltaBelow = [];
  fiftyPercentile.forEach((element, index) => {
    deltaBelow[index] = fiftyPercentile[index] - tenPercentile[index];
  });

  chart = {
    x: xValues,
    y: fiftyPercentile,
    name: 'Control',
    error_y: {
      type: 'data',
      symmetric: false,
      array: deltaAbove,
      arrayminus: deltaBelow
    },
    type: 'bar'
  };

  charts = [chart];

  layout = {
    margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }, //hide title area
    dragmode: false,
    xaxis: {
      title: 'Job Category',
      automargin: true,
      titlefont: {
        family: 'Arial, sans-serif',
        size: 18,
        color: 'lightgrey'
      }
    },
    yaxis: {
      title: 'Number of People',
      automargin: true,
      titlefont: {
        family: 'Arial, sans-serif',
        size: 18,
        color: 'lightgrey'
      }
    },
    autosize: true,
    width: '90%'
  };

  config = {
    responsive: true,
    displayModeBar:false,
    scrollZoom: false
  }

  Plotly.newPlot(elemID, charts, layout, config);

  console.log("end ageBinChart");
}

function homeZipcodeChart(elemID) {
  console.log("begin ageBinChart");
  console.log({ 'elemID': elemID });

  data = plotData[elemID];

  zipcodes = Object.keys(data);
  zipcodes.forEach(zipcode => {
    if (Object.keys(data[zipcode]).length == 0) {
      delete data[zipcode];
    }
  });
  zipcodes = Object.keys(data);

  sortedZipcodes = zipcodes.sort(function(a,b){
    return data[b]['50 percentile'] - data[a]['50 percentile']
  });

  selectedData = [];
  sortedZipcodes.forEach(zipcode => {
    selectedData.push(data[zipcode]);
  });
  console.log({ 'selectedData': selectedData });

  xValues = sortedZipcodes;

  base = selectedData;
  console.log({ 'base': base });

  tenPercentile = [];
  for (key in base) {
    tenPercentile.push(base[key]['10 percentile']);
  }

  fiftyPercentile = [];
  for (key in base) {
    fiftyPercentile.push(base[key]['50 percentile']);
  }

  nintyPercentile = [];
  for (key in base) {
    nintyPercentile.push(base[key]['90 percentile']);
  }


  deltaAbove = [];
  fiftyPercentile.forEach((element, index) => {
    deltaAbove[index] = nintyPercentile[index] - fiftyPercentile[index];
  });

  deltaBelow = [];
  fiftyPercentile.forEach((element, index) => {
    deltaBelow[index] = fiftyPercentile[index] - tenPercentile[index];
  });

  chart = {
    x: xValues,
    y: fiftyPercentile,
    name: 'Control',
    error_y: {
      type: 'data',
      symmetric: false,
      array: deltaAbove,
      arrayminus: deltaBelow
    },
    type: 'bar'
  };

  charts = [chart];

  layout = {
    margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }, //hide title area
    dragmode: 'zoom',
    xaxis: {
      title: 'Home Zip Code',
      type: 'category',
      automargin: true,
      titlefont: {
        family: 'Arial, sans-serif',
        size: 18,
        color: 'lightgrey'
      }
    },
    yaxis: {
      title: 'Number of People',
      automargin: true,
      fixedrange: true,
      titlefont: {
        family: 'Arial, sans-serif',
        size: 18,
        color: 'lightgrey'
      }
    },
    autosize: true,
    width: '90%'
  };

  config = {
    responsive: true,
    modeBarButtonsToRemove: [
      'select2d', 'lasso2d',
      'toggleSpikelines', 'autoScale2d',
      'hoverClosestCartesian', 'hoverCompareCartesian'
    ],
    scrollZoom: false,
    displayModeBar: true
  }

  Plotly.newPlot(elemID, charts, layout, config);

  console.log("end ageBinChart");
}

function commuteTypeChart(elemID) {
  console.log("begin ageBinChart");
  console.log({ 'elemID': elemID });

  data = plotData[elemID];

  xValues = Object.keys(data);
  console.log({ 'xValues': xValues });

  base = Object.values(data);
  console.log({ 'base': base });

  tenPercentile = [];
  for (key in base) {
    tenPercentile.push(base[key]['10 percentile']);
  }

  fiftyPercentile = [];
  for (key in base) {
    fiftyPercentile.push(base[key]['50 percentile']);
  }

  nintyPercentile = [];
  for (key in base) {
    nintyPercentile.push(base[key]['90 percentile']);
  }


  deltaAbove = [];
  fiftyPercentile.forEach((element, index) => {
    deltaAbove[index] = nintyPercentile[index] - fiftyPercentile[index];
  });

  deltaBelow = [];
  fiftyPercentile.forEach((element, index) => {
    deltaBelow[index] = fiftyPercentile[index] - tenPercentile[index];
  });

  chart = {
    x: xValues,
    y: fiftyPercentile,
    name: 'Control',
    error_y: {
      type: 'data',
      symmetric: false,
      array: deltaAbove,
      arrayminus: deltaBelow
    },
    type: 'bar'
  };

  charts = [chart];

  layout = {
    margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }, //hide title area
    dragmode: false,
    xaxis: {
      title: 'Commute Type',
      automargin: true,
      titlefont: {
        family: 'Arial, sans-serif',
        size: 18,
        color: 'lightgrey'
      }
    },
    yaxis: {
      title: 'Number of People',
      automargin: true,
      titlefont: {
        family: 'Arial, sans-serif',
        size: 18,
        color: 'lightgrey'
      }
    },
    autosize: true,
    width: '90%'
  };

  config = {
    responsive: true,
    displayModeBar:false,
    scrollZoom: false
  }

  Plotly.newPlot(elemID, charts, layout, config);

  console.log("end ageBinChart");
}

// SIMULATION-2: BUILDING USAGE

function campusTimePlot(label, elemID) {
  console.log("begin campusTimePlot");
  console.log({ 'label': label, 'elemID': elemID });

  data = plotData[elemID];

  // per_hour
  xValues = Object.keys(data['per_hour']['all_buildings']);
  console.log({'xValues': xValues});

  values = Object.values(data['per_hour']['all_buildings']);
  console.log({'values': values});

  yValues = [];
  values.forEach(value => {
    yValues.push(value['10 percentile']);
  });
  console.log({'yValues per_hour 10%': yValues});

  var trace1 = {
    x: xValues,
    y: yValues,
    line: {width: 0},
    marker: {color: "444"},
    mode: "lines",
    // stackgroup: 'a',
    name: "Hourly Counts: 10%",
    showlegend: false,
    type: "scatter",
    hovertemplate: '%{y:} (hourly counts: 10%)<extra></extra>'
  };

  yValues = [];
  values.forEach(value => {
    yValues.push(value['50 percentile']);
  });
  console.log({'yValues per_hour 50%': yValues});

  var trace2 = {
    x: xValues,
    y: yValues,
    // stackgroup: 'a',
    fill: "tonexty",
    fillcolor: "rgba(255, 140, 0, 0.3)",
    line: {color: "rgb(255, 50, 50)"},
    mode: "lines",
    name: "Hourly Counts",
    type: "scatter",
    hovertemplate: '%{y:} (hourly counts: median)<extra></extra>'
  };

  yValues = [];
  values.forEach(value => {
    yValues.push(value['90 percentile']);
  });
  console.log({'yValues per_hour 90%': yValues});

  var trace3 = {
    x: xValues,
    y: yValues,
    // stackgroup: 'a',
    fill: "tonexty",
    fillcolor: "rgba(255, 140, 0, 0.3)",
    line: {width: 0},
    marker: {color: "444"},
    mode: "lines",
    name: "Hourly Counts: 90%",
    showlegend: false,
    type: "scatter",
    hovertemplate: '%{y:} (hourly counts: 90%)<extra></extra>'

  }

  // per_day
  xValues = Object.keys(data['per_day']['all_buildings']);
  console.log({'xValues': xValues});

  values = Object.values(data['per_day']['all_buildings']);
  console.log({'values': values});

  yValues = [];
  values.forEach(value => {
    yValues.push(value['10 percentile']);
  });
  console.log({'yValues per_day 10%': yValues});

  var trace4 = {
    x: xValues,
    y: yValues,
    line: {width: 0},
    marker: {color: "444"},
    mode: "lines",
    // stackgroup: 'a',
    name: "Daily Counts: 10%",
    showlegend: false,
    type: "scatter",
    hovertemplate: '%{y:} (daily counts: 10%)<extra></extra>'
  };

  yValues = [];
  values.forEach(value => {
    yValues.push(value['50 percentile']);
  });
  console.log({'yValues per_day 50%': yValues});

  var trace5 = {
    x: xValues,
    y: yValues,
    // stackgroup: 'a',
    fill: "tonexty",
    fillcolor: "rgba(255, 140, 0, 0.3)",
    line: {color: "rgb(255, 50, 50)"},
    mode: "lines",
    name: "Daily Counts",
    type: "scatter",
    hovertemplate: '%{y:} (daily counts: median)<extra></extra>'
  };

  yValues = [];
  values.forEach(value => {
    yValues.push(value['90 percentile']);
  });
  console.log({'yValues per_day 90%': yValues});

  var trace6 = {
    x: xValues,
    y: yValues,
    // stackgroup: 'a',
    fill: "tonexty",
    fillcolor: "rgba(255, 140, 0, 0.3)",
    line: {width: 0},
    marker: {color: "444"},
    mode: "lines",
    name: "Daily Counts: 90%",
    showlegend: false,
    type: "scatter",
    hovertemplate: '%{y:} (daily counts: 90%)<extra></extra>'
  }

  var charts = [trace1, trace2, trace3, trace4, trace5, trace6];

  var layout = {
    margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }, //hide title area
    dragmode: 'zoom',
    autosize: true,
    showlegend: true,
    xaxis: {
      title: 'Time',
      automargin: true,
      titlefont: {
        family: 'Arial, sans-serif',
        size: 18,
        color: 'lightgrey'
      },
      gridcolor: "rgb(255,255,255)",
      showgrid: true,
      showline: false,
      showticklabels: true,
      tickcolor: "rgb(127,127,127)",
      ticks: "outside",
      zeroline: false
    },
    yaxis: {
      title: label,
      automargin: true,
      fixedrange: true,
      titlefont: {
        family: 'Arial, sans-serif',
        size: 18,
        color: 'lightgrey'
      },
      gridcolor: "rgb(255,255,255)",
      showgrid: true,
      showline: false,
      showticklabels: true,
      tickcolor: "rgb(127,127,127)",
      ticks: "outside",
      zeroline: false,
      exponentformat: 'e',
      showexponent: 'all'
    }
  };

  config = {
    responsive: true,
    modeBarButtonsToRemove: [
      'select2d', 'lasso2d',
      'toggleSpikelines', 'autoScale2d',
      'hoverClosestCartesian', 'hoverCompareCartesian'
    ],
    scrollZoom: false,
    displayModeBar: true
  };

  Plotly.newPlot(elemID, charts, layout, config);
  console.log("end buildingTimePlot");
}

function buildingTimePlot(label, building, elemID) {
  console.log("begin buildingTimePlot");
  console.log({ 'label': label, 'building': building, 'elemID': elemID });

  data = plotData[elemID];

  // per_hour
  xValues = Object.keys(data['per_hour'][building]);
  console.log({'xValues': xValues});

  values = Object.values(data['per_hour'][building]);
  console.log({'values': values});

  yValues = [];
  values.forEach(value => {
    yValues.push(value['10 percentile']);
  });
  console.log({'yValues per_hour 10%': yValues});

  var trace1 = {
    x: xValues,
    y: yValues,
    line: {width: 0},
    marker: {color: "444"},
    mode: "lines",
    // stackgroup: 'a',
    name: "Hourly Counts: 10%",
    showlegend: false,
    type: "scatter",
    hovertemplate: '%{y:} (hourly counts: 10%)<extra></extra>'
  };

  yValues = [];
  values.forEach(value => {
    yValues.push(value['50 percentile']);
  });
  console.log({'yValues per_hour 50%': yValues});

  var trace2 = {
    x: xValues,
    y: yValues,
    // stackgroup: 'a',
    fill: "tonexty",
    fillcolor: "rgba(255, 140, 0, 0.3)",
    line: {color: "rgb(255, 50, 50)"},
    mode: "lines",
    name: "Hourly Counts",
    type: "scatter",
    hovertemplate: '%{y:} (hourly counts: median)<extra></extra>'
  };

  yValues = [];
  values.forEach(value => {
    yValues.push(value['90 percentile']);
  });
  console.log({'yValues per_hour 90%': yValues});

  var trace3 = {
    x: xValues,
    y: yValues,
    // stackgroup: 'a',
    fill: "tonexty",
    fillcolor: "rgba(255, 140, 0, 0.3)",
    line: {width: 0},
    marker: {color: "444"},
    mode: "lines",
    name: "Hourly Counts: 90%",
    showlegend: false,
    type: "scatter",
    hovertemplate: '%{y:} (hourly counts: 90%)<extra></extra>'
  }

  // per_day
  xValues = Object.keys(data['per_day'][building]);
  console.log({'xValues': xValues});

  values = Object.values(data['per_day'][building]);
  console.log({'values': values});

  yValues = [];
  values.forEach(value => {
    yValues.push(value['10 percentile']);
  });
  console.log({'yValues per_day 10%': yValues});

  var trace4 = {
    x: xValues,
    y: yValues,
    line: {width: 0},
    marker: {color: "444"},
    mode: "lines",
    // stackgroup: 'a',
    name: "Daily Counts: 10%",
    showlegend: false,
    type: "scatter",
    hovertemplate: '%{y:} (daily counts: 10%)<extra></extra>'
  };

  yValues = [];
  values.forEach(value => {
    yValues.push(value['50 percentile']);
  });
  console.log({'yValues per_day 50%': yValues});

  var trace5 = {
    x: xValues,
    y: yValues,
    // stackgroup: 'a',
    fill: "tonexty",
    fillcolor: "rgba(255, 140, 0, 0.3)",
    line: {color: "rgb(255, 50, 50)"},
    mode: "lines",
    name: "Daily Counts",
    type: "scatter",
    hovertemplate: '%{y:} (daily counts: median)<extra></extra>'
  };

  yValues = [];
  values.forEach(value => {
    yValues.push(value['90 percentile']);
  });
  console.log({'yValues per_day 90%': yValues});

  var trace6 = {
    x: xValues,
    y: yValues,
    // stackgroup: 'a',
    fill: "tonexty",
    fillcolor: "rgba(255, 140, 0, 0.3)",
    line: {width: 0},
    marker: {color: "444"},
    mode: "lines",
    name: "Daily Counts: 90%",
    showlegend: false,
    type: "scatter",
    hovertemplate: '%{y:} (daily counts: 90%)<extra></extra>'
  }

  var charts = [trace1, trace2, trace3, trace4, trace5, trace6];

  var layout = {
    margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }, //hide title area
    dragmode: 'zoom',
    autosize: true,
    showlegend: true,
    xaxis: {
      title: 'Time',
      automargin: true,
      titlefont: {
        family: 'Arial, sans-serif',
        size: 18,
        color: 'lightgrey'
      },
      gridcolor: "rgb(255,255,255)",
      showgrid: true,
      showline: false,
      showticklabels: true,
      tickcolor: "rgb(127,127,127)",
      ticks: "outside",
      zeroline: false
    },
    yaxis: {
      title: label,
      automargin: true,
      fixedrange: true,
      titlefont: {
        family: 'Arial, sans-serif',
        size: 18,
        color: 'lightgrey'
      },
      gridcolor: "rgb(255,255,255)",
      showgrid: true,
      showline: false,
      showticklabels: true,
      tickcolor: "rgb(127,127,127)",
      ticks: "outside",
      zeroline: false,
      exponentformat: 'e',
      showexponent: 'all'
    }
  };

  config = {
    responsive: true,
    modeBarButtonsToRemove: [
      'select2d', 'lasso2d',
      'toggleSpikelines', 'autoScale2d',
      'hoverClosestCartesian', 'hoverCompareCartesian'
    ],
    scrollZoom: false,
    displayModeBar: true
  };

  Plotly.newPlot(elemID, charts, layout, config);
  console.log("end buildingTimePlot");
}
