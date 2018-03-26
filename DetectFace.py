import cv2
import time
import requests
import os
url = 'https://dev2.xjuke.com/upload/image'
dir = 'images/'
if not os.path.exists(dir):
    os.makedirs(dir, 777)
# 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值
faceCascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while (True):
    # 读取一帧
    ret, frame = cap.read()
    if ret == False:
        exit('Error: No Camera In Computor!')
    # 转为灰度模式
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 显示帧
    # cv2.imshow('frame', gray)

    # 转为图片流
    res, arr = cv2.imencode('.jpg', frame)

    imageData = arr.tostring()

    # 命名
    name = time.strftime("%Y%m%d%H%M%S", time.localtime())
    imagepath = dir + name + '.png'

    # 写文件
    fp = open(imagepath, 'wb')
    fp.write(imageData)
    fp.close()

    image = cv2.imread(imagepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 探测图片中的人脸
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5),
        flags=5
    )

    num = len(faces)
    print("发现{0}个人脸!".format(num))
    print("")
    # 上传到服务器
    if num > 0:
        files = {'img': (imagepath, open(imagepath, 'rb'), 'image/png')}
        res = requests.post(url, files=files).json()
        print(res)
    else:
        os.remove(imagepath)
    # 每5s采集一次
    time.sleep(5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放VideoCapture对象
cap.release()
cv2.destroyAllWindows()