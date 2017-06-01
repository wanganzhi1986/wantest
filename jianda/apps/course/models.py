# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from  datetime import datetime

from django.db import models

# Create your models here.


class OrientDict(models.Model):
    name = models.CharField(max_length=50, verbose_name=u'方向名称')

    class Meta:
        verbose_name = u'课程方向'
        verbose_name_plural = verbose_name

    def __unicode__(self):
        return self.name


class CategoryDict(models.Model):
    name = models.CharField(max_length=50, verbose_name=u'类别名称')
    orient = models.ForeignKey(OrientDict, verbose_name=u'方向')

    class Meta:
        verbose_name = u'课程类别'
        verbose_name_plural = verbose_name

    def __unicode__(self):
        return self.name


class Course(models.Model):
    # 课程要和机构互相挂钩
    name = models.CharField(max_length=50, verbose_name=u'课程名')
    desc = models.CharField(max_length=300, verbose_name=u'课程描述')
    # TextField 不限制输入长度
    detail = models.TextField(verbose_name=u'课程详情')
    learn_times = models.IntegerField(default=0, verbose_name=u'学习时长(分钟数)')
    students = models.IntegerField(default=0, verbose_name=u'学习人数')
    fav_nums = models.IntegerField(default=0, verbose_name=u'收藏人数')
    image = models.ImageField(upload_to='course/%Y/%m', verbose_name=u'封面图', max_length=100)
    click_nums = models.IntegerField(default=0, verbose_name=u'点击数')
    degree = models.CharField(choices=(('cj', u'初级'), ('zj', u'中级'), ('gj', u'高级')), max_length=2, verbose_name=u'难度')
    category = models.ForeignKey(CategoryDict, verbose_name=u'类别')
    youneed_konw = models.CharField(default='', max_length=300, verbose_name=u'课前须知')
    teacher_tell = models.CharField(default='', max_length=300, verbose_name=u'老师告诉你能学什么')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    class Meta:
        verbose_name = u'课程信息'
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


# 章节信息
# 课程和章节是一对多的关系
class Lesson(models.Model):
    course = models.ForeignKey(Course, verbose_name=u'课程')
    name = models.CharField(max_length=100, verbose_name=u'章节名')
    desc = models.CharField(default='', max_length=300, verbose_name=u'章节描述')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    class Meta:
        verbose_name = u'课程章节'
        verbose_name_plural = verbose_name

    def __unicode__(self):
        return self.name

    def get_lesson_video(self):
        # 获取章节视频
        return self.video_set.all()


class Video(models.Model):
    lesson = models.ForeignKey(Lesson, verbose_name=u'章节')
    name = models.CharField(max_length=100, verbose_name=u'视频名')
    # url = models.URLField(max_length=200, verbose_name=u'访问地址', default='www.baidu.com')
    url = models.FileField(verbose_name=u'上传视频', upload_to='video/%Y/%m', max_length=100)
    learn_times = models.IntegerField(default=0, verbose_name=u'视频时长(分钟数)')
    fav_nums = models.IntegerField(default=0, verbose_name=u'收藏人数')
    praise_nums = models.IntegerField(default=0, verbose_name=u'点赞人数')
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u'添加时间')

    class Meta:
        verbose_name = u'课程视频'
        verbose_name_plural = verbose_name

    def __unicode__(self):
        return self.name


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

