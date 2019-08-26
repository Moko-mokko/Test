import cv2
import datetime
import numpy as np
import sys
import statistics
import math

class Detector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
        self.nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
        self.profile_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
        self.image_path = "test/test.jpg"
        self.before = None
        self.color = (255, 255, 255)
        self.weight_ = []
        self.current_point = []
        self.target_point = []
        self.v = []

    def face_Detection(self,img):
        image_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height,width = img.shape[:2]
        image_mask = np.zeros((height,width),np.uint8)
        self.facerect = self.face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=20, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        self.profrect = self.profile_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=20, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        Rect = []
        if len(self.facerect) == 0 & len(self.profrect) > 0:
            Rect = self.profrect
        elif len(self.facerect) > 0 & len(self.profrect) == 0:
            Rect = self.facerect
        else:
            Rect = self.Unified(self.facerect,self.profrect)
        num = len(Rect)
        speaker = []
        for i, rect in enumerate(Rect):
            #cv2.rectangle(img, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), self.color, thickness=5)
            #cv2.putText(img,"No." + str(i),(rect[0],rect[1]),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),5)
            noserect = self.nose_cascade.detectMultiScale(image_gray[rect[1]+int(rect[3]/2):rect[1]+rect[3],rect[0]:rect[0]+rect[2]], scaleFactor=1.05, minNeighbors=6, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
            if len(noserect) > 0:
                _r = noserect[0]
                #cv2.rectangle(img, (rect[0]+_r[0],rect[1]+int(rect[3]/2)+_r[1]),(rect[0]+_r[0]+_r[2],rect[1]+int(rect[3]/2)+_r[1]+_r[3]), self.color, thickness=5)
                mouthrect = self.mouth_cascade.detectMultiScale(image_gray[rect[1]+int(rect[3]/2)+int(_r[3]*2/3):rect[1]+rect[3],rect[0]:rect[0]+rect[2]], scaleFactor=1.05, minNeighbors=6, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
                """
                if len(mouthrect) > 0:
                    r = mouthrect[-1]
                    if self.Judge(image_gray[rect[1]+int(rect[3]/2)+int(_r[3]*2/3)+r[1]:rect[1]+int(rect[3]/2)+int(_r[3]*2/3)+r[1]+r[3],rect[0]+r[0]:rect[0]+r[0]+r[2]]):
                        cv2.rectangle(img, (rect[0]+r[0],rect[1]+int(rect[3]/2)+int(_r[3]*2/3)+r[1]),(rect[0]+r[0]+r[2],rect[1]+int(rect[3]/2)+int(_r[3]*2/3)+r[1]+r[3]), (255,0,0), thickness=5)
                        cv2.putText(img,"True",(rect[0]+r[0],rect[1]+int(rect[3]/2)+int(_r[3]*2/3)+r[1]),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),5)
                        speaker.append(i)
                    else:
                        cv2.rectangle(img, (rect[0]+r[0],rect[1]+int(rect[3]/2)+int(_r[3]*2/3)+r[1]),(rect[0]+r[0]+r[2],rect[1]+int(rect[3]/2)+int(_r[3]*2/3)+r[1]+r[3]), self.color, thickness=5)
                        cv2.putText(img,"False",(rect[0]+r[0],rect[1]+int(rect[3]/2)+int(_r[3]*2/3)+r[1]),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),5)
                        """
        image_mask,_output = self.Bubble_mask(image_mask,Rect)
        cv2.imshow('x_3.png',image_mask)

        #init
        if len(self.current_point) == 0 or len(self.target_point) == 0:
            self.current_point = _output.copy()
            self.target_point = _output.copy()
            self.v = [0 for i in range(len(_output))]

        for id , sb_point in enumerate(_output):
            # detected number change
            if len(self.current_point) == len(_output):
                if self.target_point[id][0] != sb_point[0] or self.target_point[id][1] != sb_point[1]:
                    self.target_point[id] = sb_point
                    self.v[id] = 0
                if self.current_point[id][0] != self.target_point[id][0] or self.current_point[id][1] != self.target_point[id][1]:
                    self.v[id] += 0.2
                    self.current_point[id][0] += (self.target_point[id][0] - self.current_point[id][0]) * self.v[id]
                    self.current_point[id][1] += (self.target_point[id][1] - self.current_point[id][1]) * self.v[id]
                cv2.rectangle(img, (self.current_point[id][1] * 50 ,self.current_point[id][0] * 50),(self.current_point[id][1] * 50 + 240 ,self.current_point[id][0] * 50+ 120), (0,255,255), thickness=-1)
            else:
                self.current_point = []
                self.target_point = []
                self.v = []

        """
        for _index in _output:
            cv2.rectangle(img, (_index[1] * 50 ,_index[0] * 50),(_index[1] * 50 + 240 ,_index[0] * 50+ 120), (0,255,255), thickness=-1)
        """
        return img,num,speaker,self.current_point

    def Bubble_mask(self,_mask,Rect):
        R = 50
        _s = math.pow(10,-10)
        _output=[]
        pix_list = []
        dir_list = []
        height,width = _mask.shape[:2]
        tmp_1 = np.zeros((height,width),np.float32)
        tmp_2 = np.zeros((height,width),np.float32)
        tmp_3 = np.zeros((height,width),np.float32)
        for rect in Rect:
            #each max_value
            _seg = [np.sqrt(rect[0]**2 + rect[1]**2),rect[1],np.sqrt(rect[1]**2+(width-(rect[0]+rect[2]))**2),
                    rect[0],width - (rect[0] + rect[2]),np.sqrt(rect[0]**2+rect[1]**2),
                    height - (rect[1] + rect[3]),np.sqrt((width-(rect[0]+rect[2]))**2+(height-(rect[1]+rect[3]))**2)]
            seg_max = np.max(_seg)
            for i , l in enumerate([[0,0],[0,height],[width,0],[width,height]]):
                pix_list.append(np.sqrt((l[1]-(rect[1] + rect[3]/2))**2+(l[0]-(rect[0] + rect[2]/2))**2))
            #width
            tmp_3[0:rect[1],0:rect[0]] = _seg[0]
            tmp_3[0:rect[1],rect[0]:rect[0]+rect[2]] = _seg[1]
            tmp_3[0:rect[1],rect[0]+rect[2]:width] = _seg[2]
            tmp_3[rect[1]:rect[1]+rect[3],0:rect[0]] = _seg[3]
            tmp_3[rect[1]:rect[1]+rect[3],rect[0]+rect[2]:width] = _seg[4]
            tmp_3[rect[1]+rect[3]:height,0:rect[0]] = _seg[5]
            tmp_3[rect[1]+rect[3]:height,rect[0]:rect[0]+rect[2]] = _seg[6]
            tmp_3[rect[1]+rect[3]:height,rect[0]+rect[2]:width] = _seg[7]

            for h in range(0,height,R):
                for w in range(0,width,R):
                    #distance
                    tmp_1[h:h+R,w:w+R] = self.tone((np.sqrt((h-(rect[1] + rect[3]/2))**2+(w-(rect[0] + rect[2]/2))**2)/(np.max(pix_list))))
                    #direction
                    u = (w+R/2) - (rect[0] + rect[2]/2)
                    v = (rect[1] + rect[3]/2) - (h+R/2)
                    u_v = np.rad2deg(np.arctan2(v,(u+_s)))
                    tmp_2[h:h+R,w:w+R] = (1 + -np.sin(np.deg2rad(u_v))) /2
                    tmp_3[h:h+R,w:w+R] = 1 - (tmp_3[h:h+R,w:w+R] / seg_max)
                    #cost function
                    _mask[h:h+R,w:w+R] = 255*(0.4 * tmp_1[h:h+R,w:w+R] + 0.35 * tmp_2[h:h+R,w:w+R] + 0.35 * tmp_3[h:h+R,w:w+R])
            _mask[rect[1]:rect[1] + rect[3],rect[0]:rect[0] + rect[2]] = 255
            hoge = self.Speech_Bubble(_mask)
            _output.append(hoge)
            _mask[hoge[0] * 50 : hoge[0] * 50 + 120 ,hoge[1] * 50 : hoge[1] * 50 + 240] = 255
        return _mask,_output

    def Speech_Bubble(self,_mask):
        S =50
        kernel_3 = np.ones((120,240),np.uint8)
        height,width = _mask.shape[:2]; fheight,fwidth = kernel_3.shape[:2]
        tmp = np.zeros((int((height-fheight)/S) + 1,int((width-fwidth)/S) + 1),np.float32)
        oh,ow = tmp.shape[:2]
        for h in range(0,oh):
            for w in range(0,ow):
                tmp[h,w]=np.sum(_mask[h*S:h*S+fheight,w*S:w*S+fwidth]*kernel_3)
        _index=np.argwhere(tmp == tmp.min())[0]
        return _index

    def tone(self,x):
        return 0.0001**(math.e**(-3*x))

    def Unified(self,face_r,profile_r):
        Rect = self.profrect
        for a in self.facerect:
            Zs=30
            Ze=30
            for b in self.profrect:
                sub_sx = np.abs(a[0]-b[0])
                sub_sy = np.abs(a[1]-b[1])
                sub_ex = np.abs(a[0]+a[2]-b[0]+b[2])
                sub_ey = np.abs(a[1]+a[3]-b[1]+b[3])
                zs = np.sqrt(sub_sx**2+sub_sy**2)
                ze = np.sqrt(sub_ex**2+sub_ey**2)
                if ze < Ze & zs < Zs:
                    Ze=ze
                    Zs=zs
                    break
            if Zs == 30 & Ze ==30:
                Rect.append(a)
        #print(Rect)
        return Rect

    def w_ave(self,W):
        w=np.sum(np.array(range(1,W + 1)).astype(np.float))
        return [n / w for n in range(1,W + 1)]

    def weight(self,flg):
        W=10
        self.weight_.append(2) if flg else self.weight_.append(-1)
        if len(self.weight_) > W:
            self.weight_.pop(0)
        if len(self.weight_) == W:
            return np.dot(np.array(self.weight_).T, self.w_ave(W)) > 0.45
        return None

    def Judge(self,seg):
        thr = cv2.threshold(seg,60,255,cv2.THRESH_BINARY)[1]
        thr = ~thr
        label = cv2.connectedComponentsWithStats(thr)

        kernel_1 = np.ones((5,5),np.uint8)
        thr = cv2.morphologyEx(thr,cv2.MORPH_CLOSE,kernel_1)

        height,width = thr.shape[:2]
        count = 0
        n = label[0]
        label_pos = label[1]
        label_list = [np.sum(label_pos == N) for N in range(1,n)]
        cv2.imshow('x_2.png',thr)
        if len(label_list) > 0:
            _c = np.sort(label_list)[-1]
            if self.before is not None and np.abs(self.before - _c) > 30:
                flg = True
            else:
                flg = False
            self.before=_c
            return self.weight(flg)
        else:
            return None

    def Web_cam(self, mirror=True, size=None):
        cap = cv2.VideoCapture(0)
        while True:
            frame = cap.read()[1]
            if mirror is True:
                frame = frame[:,::-1]
            if size is not None and len(size) == 2:
                frame = cv2.resize(frame, size)
            now = datetime.time()
            image,n,s,_point=self.face_Detection(frame)
            print(n,s)
            print(_point)
            cv2.imshow("./test/detected{}.jpg".format(now), image)
            k = cv2.waitKey(1) #1m sec
            if k == 27: #Esc
                break
        cap.release()
        cv2.destroyAllWindows()

    def Holo_cam(self, image_path):
        frame = cv2.imread(image_path)
        [h, w] = frame.shape[:2]
        now = datetime.time()
        image,n,s,_point = self.face_Detection(frame)
        cv2.imwrite("./test/detected{}.jpg".format(now), image)
        return n,s,_point,h,w

def Main(image_path):
    _d = Detector()
    return  _d.Holo_cam(image_path)

if __name__ == "__main__":
    _d = Detector()
    _d.Web_cam(mirror = False,size=(800,600))
