# -*- coding: utf-8 -*-

from jianda import settings
from datetime import date
from time import time
from qiniu import Auth
from qiniu import BucketManager
from qiniu.services.storage.uploader import put_data


def uploadFile(name,data):
    #上传文件
    """
        :param name 文件名 data 文件流
        :return 文件存储地址
    """
    try:
        print "=========uploadFile==========="
        qiniu_url = settings.QINIU_URL
        bucket_name = settings.QINIU_BUCKET_DEFAULT
        access_key=settings.QINIU_ACCESS_KEY
        secret_key=settings.QINIU_SECRET_KEY
        q = Auth(access_key, secret_key)
        token = q.upload_token(bucket_name)

        filetype=name.split('.')[-1].lower()
        new_filename="%s.%s"%(getTimeName(),filetype)
        print new_filename
        ret, info = put_data(token, new_filename, data)

        '''print "ret:%s"%ret
        print "info:%s"%info
        print info.status_code'''

        return_url=''
        if info is not None:
            if info.status_code==200:
                return_url="http://%s/%s"%(qiniu_url,ret["key"])
            else:
                print('error: %s ' % info['exception'])
        return return_url
    except:
        raise

def deleteFile(url):
    #删除文件
    """
        :param url: 文件地址
        :return: 成功与否
    """
    try:
        print "========deleteFile========="
        access_key = settings.QINIU_ACCESS_KEY
        secret_key = settings.QINIU_SECRET_KEY
        bucket_name = settings.QINIU_BUCKET_DEFAULT
        qiniu_url=settings.QINIU_URL
        url_str=str(url)
        name=url_str.replace("http://%s/"%qiniu_url,'')

        q = Auth(access_key, secret_key)
        bucket = BucketManager(q)
        ret, info = bucket.delete(bucket_name, name)
        if info.status_code==200:
            return True
        else:
            return False
    except:
        raise

def getInfo(url):
    #获取文件信息
    """
    :param url:文件地址
    :return:文件信息
    """
    access_key = settings.QINIU_ACCESS_KEY
    secret_key = settings.QINIU_SECRET_KEY
    bucket_name = settings.QINIU_BUCKET_DEFAULT

    q = Auth(access_key, secret_key)
    bucket = BucketManager(q)
    ret, info = bucket.stat(bucket_name, url)
    print(ret,info)
    assert 'hash' in ret
    return info

def getTimeName():
    #日期_时间戳 生成文件名
    return "%s_%s"% (str(date.today()),str(time()))
