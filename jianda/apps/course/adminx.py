# -*- encoding: utf-8 -*-

import xadmin
from .models import Course, CategoryDict, OrientDict, Lesson, Video
from xadmin.views import ModelAdminView, ModelFormAdminView, ListAdminView
from django.template.response import SimpleTemplateResponse, TemplateResponse


class CourseAdmin(object):
    list_display = ['name', 'desc', 'detail', 'degree', 'learn_times', 'students', 'fav_nums',
                    'click_nums', 'add_time']
    search_fields = ['name', 'desc', 'detail', 'degree', 'learn_times', 'students', 'fav_nums',
                     'click_nums']
    list_filter = ['name', 'desc', 'detail', 'degree', 'learn_times', 'students', 'fav_nums',
                   'click_nums', 'add_time']

    style_fields = {"detail": "ueditor"}
    import_excel = True


class LessonAdmin(object):
    list_display = ['name', 'desc', 'course']
    show_detail_fields = ['name']
    search_fields = []
    list_filter = []
    object_list_template = "admin/lesson.html"



class VideoAdmin(object):
    list_display = []
    search_fields = []
    list_filter = []



class LessonAdminView(ModelAdminView):

    def get(self, request):
        pass

    def post(self, request):
        pass


    def get_context(self):
        pass


    def get_media(self):
        pass


xadmin.site.register(Course, CourseAdmin)
xadmin.site.register(CategoryDict)
xadmin.site.register(OrientDict)
xadmin.site.register(Video)
xadmin.site.register(Lesson, LessonAdmin)




