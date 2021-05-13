window.onload = function() {
  console.log('window.onload');
  initializeApp();
};

window.onpopstate = function() {
  console.log('window.onpopstate');
  initializeApp();
};
