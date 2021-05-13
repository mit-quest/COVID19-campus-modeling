// random helper functions

function createUUID() {
  return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
    (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
  );
}

function getNested(obj, ...args) {
  return args.reduce((obj, level) => obj && obj[level], obj)
}

function parseQueryString(queryString) {
  console.log("begin parseQueryString");
  console.log({'queryString': queryString});

  // save query into dict
  var queryDict = {};
  queryString.substr(1).split("&").forEach(function(item) {
    queryDict[item.split("=")[0]] = item.split("=")[1]
  });
  console.log({'queryDict': queryDict});

  if (queryDict.hasOwnProperty("page")) {
    page = queryDict['page'];
  } else {
    page = 'default';
  }

  if (queryDict.hasOwnProperty("view")) {
    view = queryDict['view'];
  } else {
    view = 'default';
  }

  if (queryDict.hasOwnProperty("uuid")) {
    uuid = queryDict['uuid'];
  } else {
    uuid = 'default'
  }

  if (queryDict.hasOwnProperty("debug")) {
    debug = queryDict['debug'];
  } else {
    debug = 'false'
  }

  parameters = {'page': page, 'view': view, 'uuid': uuid, 'debug': debug};
  console.log({'parameters': parameters});
  console.log("end parseQueryString");
  return parameters;

}

// main functions

function initializeApp() {
  console.log("begin initializeApp");

  parameters = parseQueryString(window.location.search);

  globalDebug = parameters['debug'];
  requestedPage = parameters['page'];
  requestedView = parameters['view'];
  requestedUUID = parameters['uuid'];

  initializeHome();

  if (requestedUUID != globalUUID && requestedUUID != 'default') {
    console.log('NEW UUID - REQUIRES DATA RELOAD');
    globalUUID = requestedUUID;
    initializeReport();
    setView(requestedPage, requestedView, globalUUID, false);
  }

  console.log("end initializeApp");
}

function initializeHome() {
  console.log("begin initializeHome");
  console.log(requestedUUID);
  homePage = document.getElementById('div-home');

  if (requestedUUID != 'default') {
    divCurrentUUID = document.createElement('h2');
    divCurrentUUID.innerHTML = 'Current Report UUID';
    homePage.appendChild(divCurrentUUID);

    divCurrentUUID = document.createElement('p');
    divCurrentUUID.innerHTML = 'UUID: ' + requestedUUID;
    homePage.appendChild(divCurrentUUID);

    divCurrentUUID = document.createElement('div');
    divCurrentUUID.setAttribute('class', 'row');
    homePage.appendChild(divCurrentUUID);
  }

  document.getElementById('page-home').classList.remove('d-none');
  console.log("end initializeHome");

  generateHome();
}

function generateHome() {
  console.log("begin generateHome");

  Promise.all([
    getUUIDs()
  ]).then( (values) => {
    console.log({ 'values': values });

    uuids = values[0].split("<br/>");
    uuids = uuids.filter(item => !isNaN(item));
    uuids.sort(function(a,b){return Number(b)-Number(a)});
    console.log({ 'uuids': uuids });

    list = document.getElementById('uuid-list');
    uuids.forEach(uuid => {
        element = document.createElement('li');
        willWorkQ = Number(uuid) > 0;
        if (willWorkQ) {
            element.innerHTML = '<a href="../situational-awareness?page=simulation&uuid=' + uuid + '" target="_blank">' + uuid + ' </a>';
        } else {
            element.innerHTML = uuid + ' - deprecated';
        }
        list.appendChild(element);
    });
  })

  console.log("end generateHome");
}

function initializeReport() {
  console.log("begin initializeReport");
  time = Date.now();

  navbarPagesElement = document.getElementById('navbar-pages');
  console.log({'navbarPagesElement': navbarPagesElement});

  pages = document.querySelectorAll('.page');
  console.log({'pages': pages});

  pages.forEach( function(pageElement, index) {
    console.log({'index': index, 'pageElement': pageElement});

    page = pageElement.id.substring(5); //drops page- and keeps the rest
    pageTitle = pageElement.dataset.title;

    navItem = document.createElement('li');
    navItem.setAttribute('class', 'navbar-page navbar-item btn');
    navItem.id = 'navbar-page-' + page;
    navItem.innerHTML = pageTitle;
    navItem.setAttribute('onclick', "setView('" + page + "', 'default', globalUUID, true)");
    navbarPagesElement.appendChild(navItem);

    if (pageElement.classList.contains('page-with-submenu')) {

      submenus = document.querySelectorAll('#controls-' + page + ' > div > button');
      console.log({'submenus': submenus});

      submenus.forEach((submenu, index) => {
        submenuNumber = index+1;
        submenu.setAttribute('onclick', "setView('" + page + "', '" + submenuNumber + "', globalUUID, true)");
        submenu.id = 'select-view-' + page + '-' + submenuNumber;
        submenu.classList.add('select-view-' + page);

        viewDIV = document.createElement('div');
        viewDIV.classList.add('col-12');
        questionDIV = document.createElement('div');
        questionDIV.classList.add('row');
        questionDIV.classList.add('view-' + page);
        questionDIV.id = 'view-' + page + '-' + submenuNumber;
        viewDIV.appendChild(questionDIV);
        pageElement.appendChild(viewDIV);
      });

      setView(page, 'default', globalUUID, false);

    };

  });

  time = Date.now() - time;
  console.log({'timing of initializeReport': time});
  console.log("end initializeReport");

  generateReport();

}

function generateReport() {

  console.log("begin generateReport");
  time1 = Date.now();

  if (globalDebug != 'true') {
    activateModal('preloader');
  }

  Promise.all([
    getData()
  ]).then( (values) => {

      console.log("getData finished and generateReport can continue");
      time2 = Date.now();
      loggedIn = true;

      // setup infrastructure
      if (loggedIn) {

        var data = 'declare local';

        allData = values[0];
        console.log({ 'allData': allData });

        // manage failure modes
        dataLoadError = false;
        dataLoadErrorMessage = '';

        if (allData == false) {
          dataLoadError = true;
          dataLoadErrorMessage = 'failed to parse api-server data';
        }

        if (allData != false && 'failure-message' in allData) {
          dataLoadError = true;
          dataLoadErrorMessage = allData['failure-message'];
        }

        if (dataLoadError == true) {

          console.log(dataLoadErrorMessage);

          setTimeout(function(){
            setTimeout(function(){
              deactivateModal("preloader");
            }, 250);
            setTimeout(function(){
              activateModal("data-load-error");
            }, 250);
          }, 250);

          return false;

        }
      }

      // metadata
      if (loggedIn) {
        id = getNested(allData, 'metadata', 'analysis_id');
        if ( id != undefined ) {
          displayMenuText('Analysis ID = ' + id + '<br/>' + globalUUID);
        }
      }
      console.log({'plotData': plotData});

      // could add peaks from old code
      if (loggedIn) {
        questionID = 'simulation-1';

        addSectionBreak(questionID, 'Individual Building Stats', '');

        if (
          'results' in allData &&
          'simulation' in allData['results']
        ) {

          linkedPlotNames = [
            'building-visits-time-plot',
            'building-inflow-time-plot',
            'building-outflow-time-plot'
          ];

          linkedControls = [
            'control-' + questionID + '-' + 'building-visits-time-plot',
            'control-' + questionID + '-' + 'building-inflow-time-plot',
            'control-' + questionID + '-' + 'building-outflow-time-plot',
          ];

          hourly = getNested(allData, 'results', 'simulation', 'occupancy', 'per_hour');
          daily = getNested(allData, 'results', 'simulation', 'peak_daily_occupancy', 'per_day');
          plotName = 'building-visits-time-plot';
          plotData[questionID + '-' + plotName] = {'per_hour': hourly, 'per_day': daily};
          initializeDataViz('col-6', questionID, plotName, linkedPlotNames, linkedControls);

          hourly = getNested(allData, 'results', 'simulation', 'inflow', 'per_hour');
          daily = getNested(allData, 'results', 'simulation', 'peak_daily_inflow', 'per_day');
          plotName = 'building-inflow-time-plot';
          plotData[questionID + '-' + plotName] = {'per_hour': hourly, 'per_day': daily};
          initializeDataViz('col-6', questionID, plotName, linkedPlotNames, linkedControls);

          data = getNested(allData, 'results', 'simulation', 'outflow');
          plotName = 'building-outflow-time-plot';
          plotData[questionID + '-' + plotName] = data;
          initializeDataViz('col-6', questionID, plotName, linkedPlotNames, linkedControls);

          document.getElementById(linkedControls[0]).onchange();

        }


      }
      console.log({'plotData': plotData});

      // CLOSE PRELOADER MODAL
      setTimeout(function(){ deactivateModal("preloader"); }, 1000);

      time3 = Date.now();
      console.log({'timing of generateReport': time3 - time1});
      console.log({'timing of promises': time2 - time1});
      console.log({'timing of plotting': time3 - time2});
      console.log("end generateReport");
  })

  return 'done';
}

// data retrieval

async function getUUIDs() {

  console.log("begin getUUIDs");

  // this tells server to pull data from data broker on local or remote
  var getUrl = window.location;
  var url = getUrl.protocol + "//" + getUrl.host + "/get-uuids/situational-awareness";
  console.log({"url": url});
  var response = await fetch(url,
      {
          method: 'GET',
          headers: {
              'Content-Type': 'text/plain'
          }
      }
  );
  var fromServer = await response.text();
  console.log({"fromServer": fromServer});

  console.log("end getUUIDs");
  return fromServer

}

async function getData() {

  console.log("begin getData");
  var time = Date.now();

  var getUrl = window.location;
  var url = getUrl.protocol + "//" + getUrl.host + "/get-data/situational-awareness";
  console.log({"url": url});

  // use POST in case you want to pass password for verification, this is not done in this app
  const form = new FormData();
  form.append('uuid', globalUUID);
  var response = await fetch(url, {method: 'POST', body: form});
  var body = await response.text();
  console.log({"body": body});

  try {
    parsedJSON = JSON.parse(body);
  } catch (error) {
    parsedJSON = false;
  }

  time = Date.now() - time;
  console.log({'parsedJSON': parsedJSON});
  console.log({'timing of getData': time});
  return parsedJSON;

}

// NAVIGATE APP

function resolveDefaultViews(page, view) {
  console.log("begin resolveDefaultViews");
  console.log({'page': page, 'view': view});

  if (page == 'default') {
    page = 'home';
    view = '1';
  }

  if (view == 'default') {
    view = '1';
  }

  resolved = {'page': page, 'view': view};
  console.log({'resolved': resolved});
  console.log("end resolveDefaultViews");
  return resolved;
}

function setPage(requestedPage, uuid, addToHistory) {
  console.log("begin setPage");
  console.log({'requestedPage': requestedPage, 'addToHistory': addToHistory});

  parameters = resolveDefaultViews(requestedPage, 'default');
  page = parameters['page'];

  // update URL
  if (addToHistory) {
    url = '?page=' + page + '&uuid=' + uuid;
    console.log({'url': url});
    window.history.pushState(null, '', url);
  }

  // navbar indicators
  console.log("deactivate navbar-page");
  document.querySelectorAll('.navbar-page').forEach(function (element, index) {
    element.classList.remove('active');
  });
  elemID = 'navbar-page-' + page;
  console.log({'elemID': elemID});
  document.getElementById(elemID).classList.add('active');

  // Show content, in reverse order to hide flicker of content in background
  console.log('hide and deactivate page');
  document.querySelectorAll('.page').forEach(function (element, index) {
    element.classList.remove('active');
    element.classList.add('d-none');
  });

  console.log('show selected page');
  console.log({'selected page': page});
  document.getElementById('page-' + page).classList.remove('d-none');
  document.getElementById('page-' + page).classList.add('active');

  // hide questions then slide them down
  if (addToHistory) {
    $('.question-selector').slideUp(0);
    $('.question-selector').slideDown(500);
  }

  // kludgy fix to enforce proper graphic scaling
  window.dispatchEvent(new Event('resize'));

  console.log("end setPage");

}

function setView(requestedPage, requestedView, uuid, addToHistory) {
  console.log("begin setView");
  console.log({'requestedPage': requestedPage, 'requestedView': requestedView, 'uuid': uuid, 'addToHistory': addToHistory});

  parameters = resolveDefaultViews(requestedPage, requestedView);
  page = parameters['page'];
  view = parameters['view'];

  hasSubmenus = document.getElementById('page-' + page).classList.contains('page-with-submenu');
  if (!hasSubmenus) {
    setPage(page, uuid, addToHistory);
    document.title = globalTitle + ' - ' + page;
    console.log("end setView early for lack of submenus");
    return 1;
  } else {
    setPage(page, uuid, false);
    document.title = globalTitle + ' - ' + page + ' - ' + view;
  }

  // update URL
  if (addToHistory) {
    url = '?page=' + page + '&view=' + view + '&uuid=' + uuid;
    console.log({'url': url});
    window.history.pushState(null, '', url);
  }

  console.log("select-view");
  document.querySelectorAll('.select-view-' + page).forEach(function (element, index) {
    element.classList.remove('active');
  });
  navBarView = document.getElementById('select-view-' + page + '-' + view);
  console.log({'navBarView': navBarView});
  if (navBarView != null) {
    navBarView.classList.add('active');
  }

  console.log('hide everything');

  document.querySelectorAll('.view-' + page).forEach(function (element, index) {
    element.classList.remove('active');
    element.classList.add('d-none');
  });

  console.log('show selected view');

  document.getElementById('view-' + page + '-' + view).classList.add('active');
  document.getElementById('view-' + page + '-' + view).classList.remove('d-none');

  // set view titles
  viewSelection = document.getElementById('select-view-' + page + '-' + view);
  if (viewSelection != null) {
    title = viewSelection.innerHTML;
    document.getElementById('title-' + page).innerHTML = title;
  }

  // kludgy fix to enforce proper graphic scaling
  window.dispatchEvent(new Event('resize'));

  console.log("end setView");

}

function activateModal(elemID) {
  $('#' + elemID).modal({'show': true, 'backdrop': 'static'});
}

function deactivateModal(elemID) {
  $('#' + elemID).modal('hide');
}
