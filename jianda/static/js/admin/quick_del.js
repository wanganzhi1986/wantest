(function($){

  var DeletePop = function(element){
    this.$element = $(element);
    this.del_url = this.$element.data('del-url');
    this.refresh_url = this.$element.data('refresh-url')
    this.content = this.$element.data('content')
    this.$element.on('click', $.proxy(this.click, this));
  };

  DeletePop.prototype = {
      constructor: DeletePop,

      click: function(e){
        e.stopPropagation();
        e.preventDefault();
        var modal = $('#detail-modal-id');
        var el = this.$element;
        var self = this
        if(!modal.length){
          modal = $('<div id="detail-modal-id" class="modal fade detail-modal" role="dialog"><div class="modal-dialog"><div class="modal-content">'+
            '<div class="modal-header"><button type="button" class="close" data-dismiss="modal" aria-hidden="true">&times;</button><h4 class="modal-title">'+ 
            el.attr('title') +'</h4></div><div class="modal-body"></div>'+
            '<div class="modal-footer">'+
            '<a class="btn btn-primary btn-submit">确定</a>'+
            '<button class="btn btn-default" data-dismiss="modal" aria-hidden="true">关闭</button>');
          $('body').append(modal);
        }
        modal.find('.modal-title').html(el.attr('title'));

        modal.find('.modal-body').html(el.data('content'));
        modal.find('.btn-submit').click(function(){
        $.ajax({
        url: self.del_url,
        type: "POST",
        beforeSend: function(xhr, settings) {
            xhr.setRequestHeader("X-CSRFToken", $.getCookie('csrftoken'));
        },
        success: function(){
            modal.modal('hide');
            window.location.reload()
        }
      })

    })
        modal.modal();
      }
  };

  $.fn.del = function () {
    return this.each(function () {
      var $this = $(this), data = $this.data('del');
      if (!data) {
          $this.data('del', (data = new DeletePop(this)));
      }
    });
  };

  $(function(){
    $('.del-handler').del();
  });

})(jQuery);


