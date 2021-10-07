import argparse
from posixpath import join
import sys
import time
import io
from pathlib import Path
from multiprocessing import Process
import multiprocessing
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,LoadSteaming
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from PIL import Image
import random
import requests
import json
import urllib
import os
import cv2
import numpy as np
import random
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def getlistcamid(limit:int):
    listcamid=[]
    url = "https://ovp.sohatv.vn/api/encode/camera/list"
    headers = {
        'Authorization': "Bearer debug",
        'X-OVP-APP': "lppx3dz4yw07xzh7fhaxb7ehlpn5b2sg"
        }
    response = requests.request("GET", url, headers=headers, params ={'limit':f'{limit}'})
    # print(response.text)
    pathimage="/home/quang/request/images/"
    d = json.loads(response.text)
    # print(d["data"]["items"])
    for ihh in d["data"]["items"]:
        
        # cameraId=[ihh["cameraId"]]
        listcamid.append(ihh["cameraId"])
    return listcamid
def getlistCamEvevator():
    return [10080434, 10080444  ,10080317 ,10080468, 10080467, 10080455 , 10080454  ,10080453  ,10080436, 10080445 ,10080451,10080449,  10080447,10080438, 10080444, 10080437, 10080446 ,10080440, 10080450 ,10080441, 10080415, 10080414, 10080413,10080411 ,10080410 ,10080409 ,10080408 ,10080407 ,10080406, 10080405, 10080404 ,10080400, 10080399, 10080317 ]
def GetListVideoInCameraId(cameraId:int,limit:int):
    listvideo=[]
    cameraId1=[cameraId]
    url1="https://ovp.sohatv.vn/api/encode/video/list"
    headers = {
    'Authorization': "Bearer debug",
    'X-OVP-APP': "lppx3dz4yw07xzh7fhaxb7ehlpn5b2sg"
    }
    # cameraId=[i["cameraId"]]
  
    response1 = requests.request("GET", url1, headers=headers, params ={'cameraId':f'{cameraId1}','limit':f'{limit}'})

    d2 = json.loads(response1.text)
    for j in d2["data"]["items"]:
        
        CURL_video=j["cdnURL"]
        time_create=j["createdAt"]
        timestart=j["startTime"]
        endtime=j["endTime"]
        listvideo.append(j)
    return listvideo

def showFile():
    path="video4/"
    
    forged=os.listdir(path)
    for i in range(len(forged)):
        content1=[]
        pathlist=os.path.join(path,forged[i])
        forged1=os.listdir(pathlist)
        for j in range(len(forged1)):
            if "txt" in forged1[j]:
                pathlist1=os.path.join(pathlist,forged1[j])
                with open(pathlist1,"r") as fRead:
                    content=fRead.read()
                    if content !=None:
                        content1.append(str(content))
         
    return content1



def getmodel():
    weights='weights/best.pt'  # model.pt path(s)
     # file/dir/URL/glob, 0 for webcam
    imgsz=640  # inference size (pixels)
    conf_thres=0.3  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    device='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    half=False  # use FP16 half-precision inferenc
   
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # model = attempt_load(weights, map_location="cpu")  
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    
    if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        # model(torch.zeros(1, 3, imgsz, imgsz).to("cpu").type_as(next(model.parameters())))  # run once   
    return model,device
def get_predict(model,img0):
    imgsz=640  # inference size (pixels)
    conf_thres=0.3  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    device='cpu'
    line_thickness=3 # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    half=False  # use FP16 half-precision inferenc
    hide_labels=False
    set_logging()
    names = model.module.names if hasattr(model, 'module') else model.names 
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    hide_conf=False
    input_image = Image.open(io.BytesIO(img0)).convert('RGB')
    input_image = np.array(input_image)
#     cv2.imwrite("image.png",input_image)
    
    # Convert RGB to BGR
    input_image = input_image[:, :, ::-1].copy()
    
    img = letterbox(input_image, 640,32)[0]
        # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)   

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]

        # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    t2 = time_synchronized()
    for i, det in enumerate(pred):  # detections per image
        path="/"
        p, s, im0 = path, '', input_image.copy()   
        imc = input_image.copy()  
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
              
                if True or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                 
        print(f'{s}Done. ({t2 - t1:.3f}s)')
        # Stream results
        if True:
#             cv2.imwrite("image1.png",im0)
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(np.uint8(im0)).convert('RGB')
            return input_image
#             im0=cv2.resize(im0,(512,512))
#             cv2.imshow(str(p), im0)
#             cv2.waitKey(0)  # 1 millisecond
    
def detect(source,model,device):
    imgsz=640  # inference size (pixels)
    conf_thres=0.3  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
     # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    half=False 
    stride = int(model.stride.max())  # model stride
#     imgsz = check_img_size(imgsz, s=stride)  # check image size
#     names = model.module.names if hasattr(model, 'module') else model.names 
    source=str(source)
    ran=random.randint(1, 99999999)
    # print(image_save_path)
    vid_path, vid_writer,vid_path1, vid_writer1 = None, None,None, None
    try:
        if True:
            print("webcam")
            # view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadSteaming(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
    except:
        return "Not found"
            
    t0 = time.time()
            # device="cpu"
    pathvideo="/home/quang/Documents/yolo123/data/video/"
    i=0 
    imglist=[]
    im0slist=[]
    im0ssave=[]
    t1 = time_synchronized()
    i=0
    time_label={"startime":0,
#                 "i_start":0,
#                 "i_end":0,
                "endtime":0,
                "status":False
               }
    list_string=""
    timeprocess=0
    timeprocessstart=0
    for path, img, im0s, vid_cap in dataset:
                # print("sourcesourcesource",source[:10])
        fps = vid_cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame=vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps1 = vid_cap.get(cv2.CAP_PROP_FPS)
#         try:
        if i <=frame-1:
#                 im0ssave.append(im0s)
                label_time=6
                if i%label_time==1 or(i==frame-1):
                    try:
                        img = torch.from_numpy(img).to(device)
                        
                    except:
                        if(time_label['endtime']==0):
                            time_label['endtime']=f'{(frame/fps1)//60}"p"{(frame/fps1)%60:.2f}"s"'
                            return time_label
                        return list_string
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    img=img.cpu()
                    imglist.append(img)    
                    im0slist.append(im0s)
                    if(len(imglist)==0):
                        if fps==0:
                            fps=timeprocessstart+(frame/fps1-timeprocessstart)/(frame-i)
                    timeprocessstart=fps
                    #print("timeprocessstart",timeprocessstart)
                label_time=48
                if i%(label_time)==1 or(i==frame-1):   
                    img = torch.cat(imglist,dim=0).to(device)
                    pred = model(img, augment=augment)[0]
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)   
                    checkdetect=False
                    for j, det in enumerate(pred):  # detections per image
                        if fps==0:
                            fps=timeprocess+(frame/fps1-timeprocess)/(frame-i)
                        timeprocess=fps
                        if(det.shape[0]>0):
                            # det[:, :4] = scale_coords(imgor.shape[2:], det[:, :4], im0.shape).round()
                            ran=random.randint(1, 9999999999999)
                            checkdetect=True
#                             print(time_label)
                            save_path=pathvideo+str(ran)+"videoperson.avi"
                            if time_label['status']==False:
#                                 time_label["i_start"]=i-label_time
                                time_label["startime"]=f'{(timeprocessstart)//60}"p"{(timeprocessstart)%60:.2f}"s"'
                                time_label["status"]=True   
#                                 list_string=list_string+"\n"+str(str(time_label))
#                                 return list_string
                                break            
                    if checkdetect==False:
                        if time_label["status"]==True:             
#                             time_label["i_end"]=i
                            time_label["endtime"]=f'{(timeprocessstart)//60}"p"{(timeprocessstart)%60:.2f}"s"'
                            print("i",i,"---",print(time_label),time_label)
                            list_string=list_string+"\n"+str(str(time_label))
                            time_label['status']=False
                    
                    if i==frame-1 and checkdetect==True:
                        print("famecuoi")
#                         time_label["i_end"]=i
                        time_label["endtime"]=frame/fps1
#                                             time_label["endtime"]=timeprocess frame/fps1
                        print("i",i,"---",print(time_label),time_label)
                    #                                 f.write(str("i"+str(i)))
                        list_string=list_string+"\n"+str(str(time_label))
                                            # f.write(str(str(time_label)+"\n"))
                
#                     im0ssave=[]
                    im0slist=[]
                    imglist=[]
#                     t2 = time_synchronized()
#                     print("timetime",t2-t1)
                i=i+1
#             else:
#                 return list_string
#         except:
#             print("Unexpected error:", sys.exc_info()[0])
#             return list_string
#             # i=i+1