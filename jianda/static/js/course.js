/**
 * Created by wangwei on 17/4/18.
 */

function getCookie(name) {
     var cookieValue = null;
     if (document.cookie && document.cookie != '') {
         var cookies = document.cookie.split(';');
         for (var i = 0; i < cookies.length; i++) {
             var cookie = jQuery.trim(cookies[i]);
             // Does this cookie string begin with the name we want?
             if (cookie.substring(0, name.length + 1) == (name + '=')) {
                 cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                 break;
             }
         }
     }
     return cookieValue;
 };

$(function () {

    $(".video-btn .btn-praise").click(function () {
        var video_id = $(this).data("praiseid");
        var $elem = $(this).find(".num");
        $.ajax({
            cache: false,
            type: "POST",
            url: "/courseadd_praise/",
            data: {'video_id': video_id},
            async: true,
            beforeSend: function (xhr, settings) {
                xhr.setRequestHeader("X-CSRFToken", getCookie('csrftoken'));
            },
            success: function (data) {
                console.log('success')
                console.log(data.nums)

                $elem.text(data.nums);
                console.log($elem.text());
                // if (data.status == 'fail') {
                //     if (data.msg == '用户未登录') {
                //         window.location.href = "{% url 'login' %}";
                //     } else {
                //         $(this).text(data.msg)
                //     }
                // } else if (data.status == 'success') {
                //     praise_num += 1;
                //     $(this).text(praise_num);
                //     // current_elem.text(data.msg)
                // }
            },
            error: function(error) {
                alert('ajax 请求失败！')
            }
        });
    });

    $(".video-btn .btn-fav").click(function () {
        var video_id = $(this).data("favid");
        var fav_num = parseInt($(this).text());
        var $elem = $(this).find(".num")
        $.ajax({
            cache: false,
            type: "POST",
            url: "/courseadd_fav/",
            data: {'video_id': video_id},
            async: true,
            beforeSend: function (xhr, settings) {
                xhr.setRequestHeader("X-CSRFToken", getCookie('csrftoken'));
            },
            success: function (data) {
                console.log('success');
                $elem.text(data.nums);
                // if (data.status == 'fail') {
                //     if (data.msg == '用户未登录') {
                //         window.location.href = "{% url 'login' %}";
                //     } else {
                //         $(this).text(data.msg)
                //     }
                // } else if (data.status == 'success') {
                //     praise_num += 1;
                //     $(this).text(praise_num);
                //     // current_elem.text(data.msg)
                // }
            },
            error: function(error) {
                alert('ajax 请求失败！')
            }
        });
    })

//   发表评论

//    $(".video-comment").on("click", ".video-comment-btn", function(){
//
//    })
    $('#js-comment-btn').click(function(){

        var content = $(this).parent().parent().find('textarea:first').val();
        console.log(content);
        var video_id = $(this).data("videoid");
        console.log(video_id);
        $.ajax({
            cache: false,
            type: "POST",
            url: "/courseadd_comment/",
            data: {'video_id': video_id, 'content': content},
            async: false,
            beforeSend: function (xhr, settings) {

                xhr.setRequestHeader("X-CSRFToken", getCookie('csrftoken'));
            },
            success: function (data) {
                var nickname = data.nickname;
                var addtime = data.addtime;
                var content = data.content;
                var evalboxs = $(this).parent().parent().parent().find(".video-comment-list ul");
                var evalbox = '<li class="comment">'+
                      '<div class="comment-avator clearfix"><a href=""><img width="40" height="40" src="image/image2.png"></a></div>'+
                      '<div class="comment-main"><div class="comment-nick clearfix"><a href="" class="l">'+ nickname +'</a></div>'+
                     '<div class="comment-content">'+ content +'</div>'+
                       '<div class="comment-btm clearfix"><span class="comment-time">'+ addtime +'</span>'+
                       '<a class="comment-praise"><i class="fa fa-thumbs-up" aria-hidden="true"></i><span>20</span></a></div></div></li>'
                evalboxs.append(evalbox);
                $(this).parent().parent().find('textarea').val('');
                window.location.reload();
            },
            error: function(error) {
                alert('ajax 请求失败！')
            }
        });

    })

});