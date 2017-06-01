# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from django.db import models

# Create your models here.


# 地区
class Region(models.Model):
    country = models.CharField(default=u'中国', max_length=10, verbose_name=u'国家')
    province = models.CharField(max_length=10, verbose_name=u'省份')
    city = models.CharField(max_length=20, verbose_name=u'市')
    county = models.CharField(max_length=20, verbose_name=u'县')


# 学校
class School(models.Model):
    name = models.CharField(max_length=30, verbose_name=u'学校名称')
    rank = models.CharField(choices=(('pt', u'普通中学'), ('zd', u'重点中学'), ('sf', u'示范中学')), verbose_name=u'学校类型')
    tel = models.CharField(max_length=20, verbose_name=u'学校电话')
    address = models.CharField(max_length=200)
    longitude = models.FloatField()
    latitude = models.FloatField()


# 班级
class ClassRoom(models.Model):
    school = models.ForeignKey(School, verbose_name=u'所属学校')
    grade = models.CharField(choices=(('gy', u'高一'), ('ge', u'高二'), ('gs', u'高三')), verbose_name=u'年级')
    room_order = models.PositiveIntegerField(verbose_name=u'班级号')
    student_num = models.PositiveIntegerField(verbose_name=u'班级人数')
    level = models.CharField(choices=(('pt', u'普通班'), ('sy', u'重点班'), ('tc', u'特长班')))







