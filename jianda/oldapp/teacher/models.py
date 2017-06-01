# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from django.db import models
from school.models import ClassRoom

# Create your models here.


class Teacher(models.Model):
    name = models.CharField(max_length=30, verbose_name=u'姓名')
    gender = models.CharField(choices=(('male', u'男'), ('female', u'女')), verbose_name='性别')
    age = models.PositiveIntegerField(verbose_name=u'年龄')
    graduate = models.CharField(max_length=50, verbose_name=u'毕业学校')
    subject = models.CharField(choices=(('yw', u'语文'), ('sx', u'数学'), ('yw', u'英语'),
                                        ('wl', u'物理'), ('hx', u'化学'), ('sw', u'生物'),
                                        ('ls', u'历史'), ('dl', u'地理'), ('zz', u'政治')
                                        ), verbose_name=u'科目')


# 教师所教班级
class TeacherClassRoom(models.Model):
    teacher = models.ForeignKey(Teacher)
    class_room = models.ForeignKey(ClassRoom)






