
;(function($){

  var QuickEditBtn = function(element, options) {
    var that = this;

    this.$btn = $(element)
    this.edit_url = this.$btn.data('edit-url')
    this.binit(element, options);
  }

  QuickEditBtn.prototype = {

     constructor: QuickEditBtn

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
          '<input type="submit" value="添加" name="_addanother" class="btn btn-primary btn-submit"></div></div></div></div>')

        $('body').append(modal)

        var self = this
//        modal.find('.modal-body').html('<h2 style="text-align:center;"><i class="fa-spinner fa-spin fa fa-large"></i></h2>')
        modal.find('.modal-body').load(this.edit_url, function(form_html, status, xhr){
          var form = $(this).find('form')
//          var action = '<div class="modal-footer"><button class="btn btn-default" data-dismiss="modal" aria-hidden="true">关闭</button>'+
//          '<button type="submit" value="添加" class="btn btn-primary"></button></div>'
//          form.append(action)
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

  $.fn.ajax_editbtn = function ( option ) {
    return this.each(function () {
      var $this = $(this), data = $this.data('ajax_editbtn');
      if (!data) {
          $this.data('ajax_editbtn', (data = new QuickEditBtn(this)));
      }
    });
  };

  $.fn.ajax_editbtn.Constructor = QuickEditBtn

  $(function(){
    $('.edit-handler').ajax_editbtn();
  });


})(jQuery)
