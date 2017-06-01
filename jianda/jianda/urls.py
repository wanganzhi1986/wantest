# -*- coding: utf-8 -*-
"""jianda URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from django.views.generic import TemplateView
from users.views import LoginView, RegisterView, ForgetPwdView
import xadmin
from django.views.static import serve
from jianda.settings import MEDIA_ROOT

urlpatterns = [
    # url(r'^admin/', admin.site.urls),
    url(r'^xadmin/', xadmin.site.urls),
    # url(r'^xadmin/', include(xadmin.site.urls, namespace='xadmin')),
    url('^$', TemplateView.as_view(template_name="index.html"), name='index'),
    url('^login/$', LoginView.as_view(), name='login'),
    url('^register/$', RegisterView.as_view(), name='register'),
    # 验证码
    url(r'^captcha/', include('captcha.urls')),
    url(r'^forget/$', ForgetPwdView.as_view(), name='forget_pwd'),

    # 学习课程
    url(r'^course', include('course.urls', namespace='course')),


    #配置上传文件的访问处理函数
    url(r'^media/(?P<path>.*)$', serve, {'document_root': MEDIA_ROOT}),
]
