

var PollingManager = (function (){

  function PollingManager ( ms ) {
    this._startTime = 0;
    this._polling = ms;
  }
  
  PollingManager.prototype.update = function (){
    var now = new Date().getTime(),
        bool = false;

    if ( ( now - this._startTime ) >= this._polling ) {

      this._startTime = now;
      bool = true;
    }
    return bool;
  };
  return PollingManager;
}());
var Main = ( function (){
  var _id, _fps;

  function _loop () {
    _id = requestAnimationFrame( _loop );

    if ( _fps.update() ) {
        // 以下にfpsに依存しループ実行したいコード
        
    };
  }
  function Main () {
      this._fps = 24;
  }
  var p = Main.prototype;
  p.setFPS = function ( fps ){
    _fps = new PollingManager( 1000 / fps );
  };
  p.start = function (){
    if ( typeof _fps === "undefined" ) {
      _fps = new PollingManager( 1000 / this._fps );
    }
    _loop();
  };
  p.stop = function (){
    cancelAnimationFrame( _id );
  };
  return Main;
}());

var main = new Main();
main.setFPS(50)
main.start