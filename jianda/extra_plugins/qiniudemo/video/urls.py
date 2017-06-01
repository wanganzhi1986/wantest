from django.conf.urls import patterns, include, url

urlpatterns = patterns('video.views',
   url(r'^$', 'index', name='index'),
   url(r'^uptoken/$', 'uptoken', name='uptoken'),
)