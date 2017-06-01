# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from datetime import datetime
from course.models import KnowledgeTopic, KnowledgePoint, Teacher


# Create your models here.


class Exam(models.Model):
    name = models.CharField(max_length=30, verbose_name=u'测试名称')
    category = models.CharField(choices=(('zjl', u'章节测试'), ('txl', u'专题测试'),
                                         ('qzl', u'期中测试'), ('qml', u'期末测试'),
                                         ('yml', u'一模测试'), ('eml', u'二模测试'),
                                         ('gkl', u'高考真题'), ('khl', u'课后测试')
                                         ), verbose_name=u'测试类型')
    exam_time = models.IntegerField(verbose_name=u'测试时长')
    subject = models.CharField(choices=(('sx', u'数学'), ('wl', u'物理')), verbose_name=u'科目')
    # 版本:必修1，必修2等
    version = models.CharField(max_length=20, verbose_name='课程版本')
    grade = models.CharField(choices=(('gy', u'高一'), ('ge', u'高二'), ('gs', u'高三')), verbose_name=u'年级')

    desc = models.CharField(max_length=300, verbose_name=u'课程描述')
    # TextField 不限制输入长度
    click_nums = models.IntegerField(default=0, verbose_name=u'点击数')
    question_nums = models.IntegerField(default=5, verbose_name=u'测试题目个数')

    def get_all_questions(self):
        return self.questions


# 问题解析视频，一个问题对应一个或多个解析视频，有些视频分成多段讲解
class ExamVideo(models.Model):
    name = models.CharField(max_length=100, verbose_name=u'视频名')
    url = models.URLField(max_length=200, verbose_name=u'访问地址', default='www.baidu.com')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')
    price = models.FloatField(default=0, verbose_name=u'价格')
    author = models.ForeignKey(Teacher, verbose_name=u"录制视频教师")

    def __unicode__(self):
        return self.name


# 测试的试题
class ExamQuestion(models.Model):
    exam = models.ForeignKey(Exam, verbose_name=u'测试')
    question = models.CharField(max_length=600)
    option1 = models.CharField(max_length=100)
    option2 = models.CharField(max_length=100)
    option3 = models.CharField(max_length=100)
    option4 = models.CharField(max_length=100)
    degree = models.FloatField(verbose_name=u'问题难度')
    score = models.PositiveIntegerField(verbose_name=u'问题分数')
    time = models.PositiveIntegerField(verbose_name=u'所用时间')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')


# 测试试题关联的知识点
class ExamPoint(models.Model):
    exam_question = models.ForeignKey(ExamQuestion, verbose_name=u'测试试题')
    knowledge_point = models.ForeignKey(KnowledgePoint, verbose_name=u'知识点')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')


# 测试试题关联的题型
class ExamTopic(models.Model):
    exam_question = models.ForeignKey(ExamQuestion, verbose_name=u'测试试题')
    knowledge_topic = models.ForeignKey(KnowledgeTopic, verbose_name=u'题型')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')


# 测试试题答案
class ExamAnswer(models.Model):
    course_question = models.ForeignKey(ExamQuestion, verbose_name=u'问题')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')


# 答案的每个点
class AnswerSheet(models.Model):
    answer = models.ForeignKey(ExamAnswer, verbose_name=u'答案')
    desc = models.TextField(verbose_name=u'答案描述')
    score = models.PositiveIntegerField(verbose_name=u"分数")
    knowledge_point = models.OneToOneField(KnowledgePoint, verbose_name=u'知识点')
    knowledge_topic = models.OneToOneField(KnowledgeTopic, verbose_name=u'题型')


class AnswerSheetDepedency(models.Model):
    source = models.ForeignKey(AnswerSheet, verbose_name=u'源知识点')
    target = models.ForeignKey(AnswerSheet, verbose_name=u'目标知识点')



