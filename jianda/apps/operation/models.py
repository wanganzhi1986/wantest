# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from course.models import Course, Video
from users.models import UserProfile
from datetime import datetime

# Create your models here.


class UserCourse(models.Model):
    user = models.ForeignKey(UserProfile, verbose_name=u'用户')
    course = models.ForeignKey(Course, verbose_name=u'课程')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    class Meta:
        verbose_name = u'用户学习过的课程'
        verbose_name_plural = verbose_name


class UserVideo(models.Model):
    user = models.ForeignKey(UserProfile, verbose_name=u'用户')
    video = models.ForeignKey(Video, verbose_name=u'视频')

    reward = models.FloatField(default=0, verbose_name=u'打赏金额')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    class Meta:
        verbose_name = u'用户学习过的视频'
        verbose_name_plural = verbose_name


class VideoComment(models.Model):
    user = models.ForeignKey(UserProfile, verbose_name=u'用户')
    video = models.ForeignKey(Video, verbose_name=u'视频')
    comment = models.CharField(max_length=300, verbose_name=u'发表评论')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    class Meta:
        verbose_name = u'用户发表的评论'
        verbose_name_plural = verbose_name


class VideoFav(models.Model):
    user = models.ForeignKey(UserProfile, verbose_name=u'用户')
    video = models.ForeignKey(Video, verbose_name=u'视频')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    class Meta:
        verbose_name = u'用户收藏的视频'
        verbose_name_plural = verbose_name


class VideoPraise(models.Model):
    user = models.ForeignKey(UserProfile, verbose_name=u'用户')
    video = models.ForeignKey(Video, verbose_name=u'视频')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    class Meta:
        verbose_name = u'用户点赞的视频'
        verbose_name_plural = verbose_name


class VideoRegard(models.Model):
    user = models.ForeignKey(UserProfile, verbose_name=u'用户')
    video = models.ForeignKey(Video, verbose_name=u'视频')
    regard = models.FloatField(verbose_name=u'打赏金额')

    class Meta:
        verbose_name = u'用户打赏'
        verbose_name_plural = verbose_name






