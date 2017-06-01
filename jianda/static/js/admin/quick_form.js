
;(function($){
  var QuickAddBtn = function(element, options) {
    var that = this;

    this.$btn = $(element)
    this.add_url = this.$btn.data('add-url')
    this.binit(element, options);
  }

  QuickAddBtn.prototype = {

     constructor: QuickAddBtn

    , binit: function(element, options){
      this.$btn.click($.proxy(this.click, this))
    }
    , click: function(e) {
      e.stopPropagation()
      e.preventDefault()
      var modal = $('#detail-modal-id');
      var el = this.$element;
      if(!this.modal){
        var modal = $('<div id="detail-modal-id" class="modal fade quick-form" role="dialog"><div class="modal-dialog"><div class="modal-content">'+
          '<div class="modal-header"><button type="button" class="close" data-dismiss="modal" aria-hidden="true">&times;</button><h3>'+
          this.$btn.attr('title') +'</h3></div><div class="modal-body"></div>'+
          '<div class="modal-footer" style="display: none;"><button class="btn btn-default" data-dismiss="modal" aria-hidden="true">关闭</button>'+
          '<a class="btn btn-primary btn-submit">添加</a></div></div></div></div>')
        $('body').append(modal)

        var self = this
//        modal.find('.modal-body').html('<h2 style="text-align:center;"><i class="fa-spinner fa-spin fa fa-large"></i></h2>')
        modal.find('.modal-body').load(this.add_url, function(form_html, status, xhr){
          var form = $(this).find('form')

          modal.find('.modal-footer').show()
          modal.find('.btn-submit').click(function(){

               form.ajaxSubmit({
                    success: function(result){
                        modal.modal('hide');
                        window.location.reload()
                    }

               })
          })

        })
        this.modal = modal
      }
      this.modal.modal();

      return false
    }

  }

  $.fn.ajax_addbtn = function ( option ) {
    return this.each(function () {
      var $this = $(this), data = $this.data('ajax_addbtn');
      if (!data) {
          $this.data('ajax_addbtn', (data = new QuickAddBtn(this)));
      }
    });
  };

  $.fn.ajax_addbtn.Constructor = QuickAddBtn

  $(function(){
    $('.form-handler').ajax_addbtn();
  });


})(jQuery)
