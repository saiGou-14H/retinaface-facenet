
from app.myface.retinaface import Retinaface
from concurrent.futures import ProcessPoolExecutor
retinaface = Retinaface()
from app.utils.utils import *
pool = ProcessPoolExecutor()
facePath = "app/face_image"


def Search(data):
    # 从请求数据读取图片信息并且对过暗过亮图片进行yuv均衡化处理
    start_time=time.time()

    img = getImg(data)
    img, imgs, old_img, silent_imgs = retinaface.detect_image(img)
    # 活体检测
    scores,imgs = retinaface.silentFace(imgs,silent_imgs)
    if scores is None:
        return "None"
    if imgs is None:
        return "None"
    user= retinaface.search_face(imgs)
    user = dict_json(user)
    print("识别结果：",user)

    print("识别耗时{:.2f}s".format(time.time() -start_time))
    return str(user)

def Add(data):
    if(data["Type"]=="BASE64S"):
        imgs = getImg(data)
        names = data["names"]
        userids = data["userids"]
        for index in range(len(imgs)):
            img, _, _, _ = retinaface.detect_image(imgs[index])
            if img is None:
                continue
            retinaface.add_face(img, userids[index], names[index])
    else:
        img=getImg(data)
        name = data["name"]
        userid = data["userid"]
        img, _, _, _ = retinaface.detect_image(img)
        if img is None:
            return str("未找到人脸")
        retinaface.add_face(img,userid,name)
    print(str("人脸录入成功"))
    return str("人脸录入成功")

def Delete(data):
    userid = data["userid"]
    retinaface.delete_face(userid)
    print("删除用户人脸成功id["+str(userid)+"]")
    return str(userid);


def Encoding(data):
    print(data)
    if ((data!=None)and(data.__contains__("path"))):
        path = data["path"]
        retinaface.encoding_face(facePath+"/"+path);
    else:
        retinaface.encoding_face(facePath)
    print("格式化人脸库成功")
    return str("格式化人脸库成功");


def Test():
    time1 = time.strftime('%Y.%m.%d.%H.%M.%S', time.localtime(time.time()))
    print(time1)
    return time1



