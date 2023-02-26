from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from flask import Blueprint, request
from app.api.Model import *
from app.app import app
from app.myface.retinaface import Retinaface

#pool = ProcessPoolExecutor(max_workers=5)#多进程
pool = ThreadPoolExecutor(max_workers=32)#多线程
retinaface = Retinaface()

Face = Blueprint('Face', __name__)

@app.route('/',methods=["POST","GET",'OPTIONS'])
def test():
    return Test()

@Face.route('/addface',methods=["POST","GET",'OPTIONS'])
def add():
    data = getPostData(request)
    return Add(data)

@Face.route('/searchface',methods=["POST","GET",'OPTIONS'])
def search():
    data = getPostData(request)
    return str(pool.submit(Search,data).result())#多线程处理高并发

@Face.route('/deleteface',methods=["POST","GET",'OPTIONS'])
def delete():
    data = getPostData(request)
    return Delete(data)

@Face.route('/encoding',methods=["POST","GET",'OPTIONS'])
def encoding():
    data = getPostData(request)
    return Encoding(data)


