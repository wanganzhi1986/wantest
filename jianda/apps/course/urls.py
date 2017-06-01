# -*- coding: utf-8 -*-

from django.conf.urls import url, include
from .views import CourseListView, CourseDetailView, CourseVideoView, AddCommentView, AddPraiseView, AddFavView


urlpatterns = [
    url('^list/$', CourseListView.as_view(), name="course_list"),
    url('^detail/(?P<course_id>\d+)/$', CourseDetailView.as_view(), name='course_detail'),
    url('^video/(?P<video_id>\d+)/$', CourseVideoView.as_view(), name='course_video'),
    url(r'^add_comment/$', AddCommentView.as_view(), name='add_comment'),
    url(r'^add_fav/$', AddFavView.as_view(), name='add_fav'),
    url(r'^add_praise/$', AddPraiseView.as_view(), name='add_praise')
]