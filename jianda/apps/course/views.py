# -*- coding: utf-8 -*-
from django.shortcuts import render
from django.views.generic.base import View
from .models import Course, CategoryDict, OrientDict, Video

from pure_pagination import Paginator, EmptyPage, PageNotAnInteger
import json
from operation.models import UserCourse, UserVideo, VideoComment, VideoRegard, VideoFav, VideoPraise
from django.http import HttpResponse
from datetime import datetime

# Create your views here.


class CourseListView(View):

    def get(self, request):
        all_course = Course.objects.all()
        all_orient = OrientDict.objects.all()
        all_category = CategoryDict.objects.all()

        orient_id = request.GET.get("orient", '')
        if orient_id:
            all_category = all_category.filter(orient_id=int(orient_id))

        category_id = request.GET.get("category", '')
        if category_id:
            all_course = all_course.filter(category_id=int(category_id))

        degree = request.GET.get("degree", '')
        print("degree is:", degree)

        if degree:
            all_course = all_course.filter(degree=degree)

        # 对课程机构进行分页
        try:
            page = request.GET.get('page', 1)
        except PageNotAnInteger:
            page = 1

        p = Paginator(all_course, 2, request=request)
        courses = p.page(page)

        return render(request, 'course-list.html', {
            "all_course": courses,
            "all_orient": all_orient,
            "all_category": all_category,
            "orient_id": orient_id,
            "category_id": category_id,
            "degree": degree
        })


class CourseDetailView(View):

    def get(self, request, course_id):

        course = Course.objects.get(id=int(course_id))
        course.click_nums += 1
        first_learn = request.GET.get('first_learn', False)
        has_learn = False
        user_course = UserCourse.objects.filter(user=request.user, course=course)
        if not user_course:
            if first_learn:
                has_learn = True
                course.students += 1
                user_course = UserCourse(user=request.user, course=course)
                user_course.save()
        else:
            has_learn = True
        # if request.user.is_authenticated():
        #
        # else:
        #     has_learn = True

        course.save()

        return render(request, 'course-detail.html', {
            'course': course,
            'has_learn': has_learn
        })


class CourseVideoView(View):

    def get(self, request, video_id):

        video = Video.objects.get(id=int(video_id))
        course = video.lesson.course
        # 用户观看了视频则对其进行保存，统计其对某个视频的观看的次数，和观看的时间
        user_video = UserVideo(user=request.user, video=video)
        user_video.save()
        # 课程评论分页
        video_comments = VideoComment.objects.filter(video=int(video_id))
        try:
            page = request.GET.get('page', 1)
        except PageNotAnInteger:
            page = 1

        p = Paginator(video_comments, 10, request=request)
        comments = p.page(page)
        return render(request, 'course-video.html', {
            "video": video,
            "course": course,
            "video_comments": comments
        }
                      )


# 添加视频收藏/取消收藏
class AddFavView(View):
    def post(self, request):
        video_id = request.POST.get('video_id', 0)
        video = Video.objects.get(id=int(video_id))
        video_fav = VideoFav.objects.filter(user=request.user, video=int(video_id))
        res = {}
        if video_fav:
            video_fav.delete()
            video.fav_nums -= 1
            if video.fav_nums < 0:
                video.fav_nums = 0
            res['status'] = 'success'
            res['msg'] = u'取消收藏'
            res['nums'] = video.fav_nums
            return HttpResponse(json.dumps(res), content_type='application/json')
            # return HttpResponse({"status":'success', 'msg': u'取消收藏', 'nums': video.fav_nums})
        else:
            video_fav = VideoFav(user=request.user, video=video)
            video_fav.save()
            video.fav_nums += 1
            res['status'] = 'success'
            res['msg'] = u'取消收藏'
            res['nums'] = video.fav_nums
            return HttpResponse(json.dumps(res), content_type='application/json')
            # return HttpResponse({'status': 'success', 'msg': u'收藏', 'nums': video.fav_nums})
        # return render(request, 'course-video.html', {"video": video})


# 添加视频收藏/取消收藏
class AddPraiseView(View):

    def post(self, request):
        video_id = request.POST.get('video_id', 0)
        video = Video.objects.get(id=int(video_id))
        video_praise = VideoPraise.objects.filter(user=request.user, video=int(video_id))
        res = {}
        if video_praise:
            video_praise.delete()
            video.praise_nums -= 1
            if video.praise_nums < 0:
                video.praise_nums = 0
            video.save()
            res['status'] = 'success'
            res['msg'] = u'取消收藏'
            res['nums'] = video.praise_nums
            return HttpResponse(json.dumps(res), content_type='application/json')
            # return HttpResponse({"status":'success', 'msg': u'取消收藏', 'nums': video.praise_nums})
        else:
            video_praise = VideoPraise(user=request.user, video=video)
            video_praise.save()
            video.praise_nums += 1
            video.save()
            res['status'] = 'success'
            res['msg'] = u'取消收藏'
            res['nums'] = video.praise_nums
            return HttpResponse(json.dumps(res), content_type='application/json')
            # return HttpResponse({'status': 'success', 'msg': u'收藏', 'nums': video.praise_nums})


# 对某个视频发表评论
class AddCommentView(View):
    def post(self, request):
        video_id = request.POST.get("video_id", 0)
        comment = request.POST.get("content", '')
        res = {}
        if video_id > 0 and comment:
            video = Video.objects.get(id=int(video_id))
            video_comment = VideoComment()
            video_comment.user = request.user
            video_comment.video = video
            video_comment.comment = comment
            video_comment.save()
            res['nickname'] = request.user.nick_name
            res['addtime'] = datetime.now().strftime('%Y-%m-%d')
            res['content'] = comment
            return HttpResponse(json.dumps(res), content_type='application/json')
        #
        # return render(request, 'course-video.html')


# 对某个视频进行打赏
class RewardVideoView(View):
    def get(self, request, video_id):
        return render(request, 'course-video.html')







