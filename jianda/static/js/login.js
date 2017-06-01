/**
 * Created by wangwei on 17/4/2.
 */

$(function () {

    function login_form_submit() {
        var $jsLoginBtn = $("#jsLoginBtn");
        var autoLogin = false;
        if ($('#jsAutoLogin').is(':checked')){
        autoLogin = true;
    }

        $.ajax({
            cache:false,
            type: 'post',
            datatype: 'json',
            url: 'index',
            data: $("#jsLoginForm").serialize() + "&autologin=" + autoLogin,
            async: false,
            beforeSend: function(XMLHttpRequest){
                $jsLoginBtn.val("正在登录中");
                $jsLoginBtn.val("disabled", "disabled")
            },
            success: function(result) {
                console.log(result)

            },

            complete: function (XMLHttpRequest) {
                $jsLoginBtn.val("登录");
                $jsLoginBtn.removeAttr("disabled")

            }

        });
    }

    //登录页面
    $('#jsLoginBtn').on('click',function(){
        login_form_submit();
    })

});