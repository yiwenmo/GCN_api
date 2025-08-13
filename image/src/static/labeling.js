(function($) {
    $.labeling_method = function( element, options ){
      var defaults = {
        onchange: function(text){
        },
      };
      var settings=$.extend({}, defaults, options);
      var __this=this;
      __this.element=element;
      __this.settings = settings;

      __this.labelAction=function(label){
        
      }
      __this.addLabel=function(x,y,w,h,text){
        var label=$('<div />').html(text).addClass('xlabel').offset({left: x,top: y}).width(w).height(h).css("position", "absolute").draggable({
          drag: function () {
            __this.update();
          },
          stop: function(){
            __this.initWithText(__this.update());
            // saveData();
          }
        }).resizable({
          resize: function(){
            __this.update();
          },
          stop: function(){
            __this.initWithText(__this.update());
            // saveData();
          }
        }).css("position", "absolute").draggable( "disable" ).show();
        $(label).mouseenter(function() {
          $(label).css("border", "2px solid #f00").draggable( "enable" );
        }).mouseleave(function() {
          $(label).css("border", "2px solid #00f").draggable( "disable" );
        });
        $(__this.element).append(label);
      };
      __this.init = function(){
        $(__this.element).dblclick(function(e){
          var w=100;
          var h=100;
          var posX = parseInt( e.pageX-$(this).offset().left )-w/2;
          var posY = parseInt( e.pageY-$(this).offset().top )-h/2;
          __this.addLabel(posX,posY,w,h,'--');
          __this.update();
          // saveData();
        });
      }
      __this.update = function() {
        var classID = 0;
        var m = "";
        
        // 確保我們有原始圖片尺寸信息
        if (!window.originalImageInfo) {
            console.error('Original image dimensions not available');
            return "";
        }
        
        const originalWidth = window.originalImageInfo.width;
        const originalHeight = window.originalImageInfo.height;
    
        $(__this.element).find('.xlabel').each(function() {
            // get display dimensions
            var displayWidth = $(this).outerWidth();
            var displayHeight = $(this).outerHeight();
            var displayLeft = $(this).offset().left - $(this).parent().offset().left;
            var displayTop = $(this).offset().top - $(this).parent().offset().top;
    
            // get current display dimensions
            var currentDisplayWidth = $(this).parent().width();
            var currentDisplayHeight = $(this).parent().height();
    
            // calculate scaling
            var scaleX = originalWidth / currentDisplayWidth;
            var scaleY = originalHeight / currentDisplayHeight;
    
            // transform to original
            var originalLeft = displayLeft * scaleX;
            var originalTop = displayTop * scaleY;
            var originalBoxWidth = displayWidth * scaleX;
            var originalBoxHeight = displayHeight * scaleY;
    
            // calculate normalized coordinates of YOLO
            var centerX = (originalLeft + originalBoxWidth/2) / originalWidth;
            var centerY = (originalTop + originalBoxHeight/2) / originalHeight;
            var normalizedWidth = originalBoxWidth / originalWidth;
            var normalizedHeight = originalBoxHeight / originalHeight;
    
            // ensure normalized coordinates are within [0, 1]
            if(centerX >= 0 && centerX <= 1 && centerY >= 0 && centerY <= 1) {
                m += sprintf("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                classID,
                centerX,
                centerY,
                normalizedWidth,
                normalizedHeight,
                originalLeft,   // show coordinates
                originalTop,
                originalBoxWidth,  // weight
                originalBoxHeight  // height
              );
                console.log(m);
            }
        });
        
        __this.settings.onchange(m);
        return m;
    };
      __this.initWithText = function(text) {
        $(__this.element).find('.xlabel').remove();
        var lines = text.split(/\r?\n/);
        
        // get current display dimensions
        var currentDisplayWidth = $(__this.element).width();
        var currentDisplayHeight = $(__this.element).height();
        
        // get original image dimensions
        const originalWidth = window.originalImageInfo.width;
        const originalHeight = window.originalImageInfo.height;
        
        // calculate scaling
        var scaleX = currentDisplayWidth / originalWidth;  // note: scaleX and scaleY are reversed
        var scaleY = currentDisplayHeight / originalHeight;
    
        for(i in lines) {
            var v = lines[i].split(/\ /);
            if(v.length >= 9) { // 9 values
                // original coordinates
                var originalLeft = parseFloat(v[5]);
                var originalTop = parseFloat(v[6]);
                var originalBoxWidth = parseFloat(v[7]);
                var originalBoxHeight = parseFloat(v[8]);
                
                // display coordinates
                var displayLeft = originalLeft * scaleX;
                var displayTop = originalTop * scaleY;
                var displayWidth = originalBoxWidth * scaleX;
                var displayHeight = originalBoxHeight * scaleY;
    
                var score = 0.6;
                __this.addLabel(displayLeft, displayTop, displayWidth, displayHeight, score);
                console.log(i,originalLeft, originalTop, originalBoxWidth, originalBoxHeight, score);
            }
        }
    };
      __this.init();
    };
    $.fn.labeling = function( options ) {
      return this.each(function() {
        if(undefined==$(this).data('labeling_method'))
          $(this).data('labeling_method', new $.labeling_method(this, options));
      });
    }

    $.fn.initWithText = function( text ) {
      if(typeof($(this).data('labeling_method'))!=='undefined')
        return $(this).data('labeling_method').initWithText(text);
      return undefined;
    }
    $.fn.update = function( text ) {
      if(typeof($(this).data('labeling_method'))!=='undefined')
        return $(this).data('labeling_method').update();
      return undefined;
    }
  })(jQuery);
  
  $(function(){
    $('.ximg').each(function(){
      $('.ximg').labeling();
    });

    $('#btn1').click(function(){
      var formData = new FormData();
      formData.append('file', $('#file')[0].files[0]);
      $.ajax({
        url : '',
        type : 'POST',
        data : formData,
        contentType: false,
        processData: false,
        success : function(data) {
          $('.ximg').width(data.width).height(data.height).css('background',`url(${data.image})`);
          $('.ximg').initWithText(data.labels);
          $('#zimg').attr('src',data.image1);
        }
      });
    });
  });(jQuery);