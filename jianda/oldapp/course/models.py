# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from django.db import models
from datetime import datetime
from teacher.models import Teacher

# Create your models here.


class Course(models.Model):
    name = models.CharField(max_length=50, verbose_name=u'课程名')

    subject = models.CharField(choices=(('sx', u'数学'), ('wl', u'物理')), verbose_name=u'科目')
    # 版本:必修1，必修2等
    version = models.CharField(max_length=20, verbose_name='课程版本')
    grade = models.CharField(choices=(('gy', u'高一'),('ge', u'高二'),('gs',  u'高三')), verbose_name=u'年级')

    desc = models.CharField(max_length=300, verbose_name=u'课程描述')
    # TextField 不限制输入长度
    detail = models.TextField(verbose_name=u'课程详情')
    # teacher = models.ForeignKey(Teacher, verbose_name=u'讲师', null=True, blank=True)
    degree = models.CharField(choices=(('cj', u'初级'), ('zj', u'中级'), ('gj', u'高级')), max_length=2, verbose_name=u'难度')
    learn_times = models.IntegerField(default=0, verbose_name=u'学习时长(分钟数)')
    students = models.IntegerField(default=0, verbose_name=u'学习人数')
    fav_nums = models.IntegerField(default=0, verbose_name=u'收藏人数')
    image = models.ImageField(upload_to='courses/%Y/%m', verbose_name=u'封面图', max_length=100)
    click_nums = models.IntegerField(default=0, verbose_name=u'点击数')
    # 课程类别
    category = models.CharField(choices=(('ky', u'新课学习'),('kh', u'课后复习'), ('zt', u'专题课程'),
                                         ('gy', u'高考一轮'),('ger', u'高考二轮'), ('gc', u'高考冲刺')),
                                verbose_name=u'课程类别')
    tag = models.CharField(default='', verbose_name=u'课程标签', max_length=10)
    youneed_konw = models.CharField(default='', max_length=300, verbose_name=u'课前须知')
    teacher_tell = models.CharField(default='', max_length=300, verbose_name=u'老师告诉你能学什么')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    class Meta:
        verbose_name = u'课程'
        verbose_name_plural = verbose_name

    def get_zj_nums(self):
        # 反向使用 ForeignKey
        # html 里可以直接使用
        # 获取该课程下面所有的章节的数量
        return self.lesson_set.all().count()

    def get_learn_users(self):
        # 得到 UserCourse 这个数据表，然后 UserCourse 有个 User 属性，
        # User 是个 ForeignKey，User.image 可以拿到 UserProfile 表 里的 image
        return self.usercourse_set.all()[:5]

    def get_course_lesson(self):
        # 获取该课程下面所有的章节
        return self.lesson_set.all()

    def __unicode__(self):
        return self.name


# 章节信息，
# 课程和章节是一对多的关系
class Lesson(models.Model):
    course = models.ForeignKey(Course, verbose_name=u'课程')
    name = models.CharField(max_length=100, verbose_name=u'章节名')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    class Meta:
        verbose_name = u'章节'
        verbose_name_plural = verbose_name

    def __unicode__(self):
        return self.name

    def get_lesson_video(self):
        # 获取章节视频
        return self.video_set.all()


class CourseBook(models.Model):
    name = models.CharField(max_length=100, verbose_name="课本名称")
    version = models.CharField(max_length=100, verbose_name=u'版本名称')


class KnowledgeModule(models.Model):
    name = models.CharField(max_length=100, verbose_name="知识模块")
    book_name = models.CharField(choices=(('b1', u'必修一'), ('b2', u'必修二'), ('bs', u'必修三'),
                                          ('b4', u'必修四'), ('b5', u'必修五'), ('x2-1', u'选修2-1'),
                                          ('x2-2', u'选修2-2'), ('x2-3', u'选修2-3'), ('x1', u'选修4-1')
                                          ), verbose_name="课本版本")


# 知识点
class KnowledgePoint(models.Model):
    module = models.ForeignKey(KnowledgeModule, verbose_name=u'知识模块')
    name = models.CharField(max_length=100, verbose_name="知识点名称")
    course_book = models.ForeignKey(CourseBook, verbose_name=u"所属的课本")
    author = models.CharField(max_length=30, verbose_name="录入人员")
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')


# 题型
class KnowledgeTopic(models.Model):
    module = models.ForeignKey(KnowledgeModule, verbose_name=u'知识模块')
    name = models.CharField(max_length=100, verbose_name="题型名称")
    course_book = models.ForeignKey(CourseBook, verbose_name=u"所属的课本")
    author = models.CharField(max_length=30, verbose_name="录入人员")
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')


# 课程微课视频
class CourseVideo(models.Model):
    name = models.CharField(max_length=100, verbose_name=u'视频名')
    url = models.URLField(max_length=200, verbose_name=u'访问地址', default='www.baidu.com')
    total_times = models.IntegerField(default=0, verbose_name=u'视频时长(分钟数)')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')
    price = models.FloatField(default=0, verbose_name=u'价格')
    author = models.ForeignKey(Teacher, verbose_name=u"录制视频教师")
    lesson = models.ForeignKey(Lesson, verbose_name=u'章节')

    def __unicode__(self):
        return self.name


#知识点和课程微课的关系表
class VideoPoint(models.Model):
    course_video = models.ForeignKey(CourseVideo, verbose_name=u"课程微课视频")
    knowledge_point = models.ForeignKey(KnowledgePoint, verbose_name=u"知识点")
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')


#题型和课程微课的关系
class VideoTopic(models.Model):
    course_video = models.ForeignKey(CourseVideo, verbose_name=u"课程微课视频")
    knowledge_topic = models.ForeignKey(KnowledgeTopic, verbose_name=u"知识点")
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')


# 问题
class CourseQuestion(models.Model):
    course_video = models.ForeignKey(CourseVideo, verbose_name=u'微课视频')
    question = models.CharField(max_length=600)
    option1 = models.CharField(max_length=100)
    option2 = models.CharField(max_length=100)
    option3 = models.CharField(max_length=100)
    option4 = models.CharField(max_length=100)
    degree = models.FloatField(verbose_name=u'问题难度')
    score = models.PositiveIntegerField(verbose_name=u'问题分数')
    time = models.PositiveIntegerField(verbose_name=u'所用时间')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')


# 问题答案
class CourseAnswer(models.Model):
    course_question = models.ForeignKey(CourseQuestion, verbose_name=u'问题')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')


# 答案的每个点
class CourseAnswerSheet(models.Model):
    answer = models.ForeignKey(CourseAnswer, verbose_name=u'答案')
    desc = models.TextField(verbose_name=u'答案描述')
    score = models.PositiveIntegerField(verbose_name=u"分数")
    knowledge_point = models.OneToOneField(KnowledgePoint, verbose_name=u'知识点')
    knowledge_topic = models.OneToOneField(KnowledgeTopic, verbose_name=u'题型')


class CourseResource(models.Model):
    course = models.ForeignKey(Course, verbose_name=u'课程')
    name = models.CharField(max_length=100, verbose_name=u'课件名')
    download = models.FileField(upload_to='course/resource/%Y/%m', verbose_name=u'资源文件', max_length=100)
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    class Meta:
        verbose_name = u'课程资源'
        verbose_name_plural = verbose_name

    def __unicode__(self):
        return self.name
