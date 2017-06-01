from django.http import HttpResponse
from django.shortcuts import render

import json
import qiniu
from qiniu import Auth, put_file

BUCKET_NAME = "YOUR_BUCKET_NAME"
ACCESS_KEY = "YOUR_ACCESS_KEY"
SECRET_KEY = "YOUR_SECRET_KEY"


def index(request):
    return render(request, 'demo/index.html')


def uptoken(request):
    q = Auth(ACCESS_KEY, SECRET_KEY)
    token = q.upload_token(BUCKET_NAME)
    data = {'uptoken': token}
    return HttpResponse(json.dumps(data), content_type="application/json")
