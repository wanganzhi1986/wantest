# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from teacher.models import Teacher
from school.models import ClassRoom
from course.models import Course
from datetime import datetime
from course.models import Video
from exam.models import Exam

# Create your models here.


class Student(models.Model):
    name = models.CharField(max_length=20, verbose_name=u'姓名')
    school = models.ForeignKey(School)
    grade = models.CharField(choices=(('gy', u'高一'),('ge', u'高二'), ('gs', u'高三')), verbose_name=u'年级')
    classroom = models.ForeignKey(ClassRoom, verbose_name="所在班级")
    teacher = models.ManyToManyField(Teacher, verbose_name="教师")


class StudentCourse(models.Model):
    user = models.ForeignKey(Student, verbose_name=u'用户')
    course = models.ForeignKey(Course, verbose_name=u'课程')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    class Meta:
        verbose_name = u'用户学习过的课程'
        verbose_name_plural = verbose_name


class StudentVideo(models.Model):
    student = models.ForeignKey(Student, verbose_name=u'学生')
    video = models.ForeignKey(Video, verbose_name=u'视频')
    start_time = models.DateTimeField(verbose_name=u'开始时间')
    end_time = models.DateTimeField(verbose_name=u"结束时间")
    pause_time = models.DateTimeField(verbose_name=u'暂停时长')
    pause_count = models.DateTimeField(verbose_name=u'暂停次数')
    comment = models.CharField(max_length=300, verbose_name=u"评论")
    score = models.IntegerField(verbose_name=u'评分')
    praise = models.FloatField(verbose_name=u'打赏')
    share = models.CharField(choices=(('qq', 'QQ'), ('wx', 'wechat'), ('wb', 'weibo')), verbose_name=u'分享')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    # # 获得观看的所有的视频
    # def get_all_videos(self):
    #     return self.video_set.all()


class StudentQuestion(models.Model):
    user = models.ForeignKey(UserProfile, verbose_name=u'学生')
    question = models.ForeignKey(Question, verbose_name=u'回答问题')
    start_time = models.DateTimeField(verbose_name=u'开始时间')
    end_time = models.DateTimeField(verbose_name=u"结束时间")


# 学生参加的测试
class StudentExam(models.Model):
    student = models.ForeignKey(Student, verbose_name=u'学生')
    exam = models.ForeignKey(Exam, verbose_name=u'测试')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')





