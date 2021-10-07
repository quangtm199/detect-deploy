from fastapi import FastAPI, File,UploadFile,Request,Form
from typing import Optional
import io
from pydantic import BaseModel
import argparse
from main123 import detect , getlistcamid ,GetListVideoInCameraId,getlistCamEvevator,showFile,getmodel,get_predict
import uvicorn
# from fakeantispoffing import get_predict, getmodel
from starlette.responses import Response
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles #new

import requests
import io
import json
import urllib
import cv2
from typing import Optional
# model ,face_detector = getmodel()
templates = Jinja2Templates(directory='templates/')
app = FastAPI()

def configure_static(app):  #new
    app.mount("/static", StaticFiles(directory="static"), name="static")
configure_static(app)

model,device = getmodel()
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt
class Item(BaseModel):
    source: str
    # description: Optional[str] = None
    # price: float
    # tax: Optional[float] = None
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

@app.post("/sysadmin/mlbigdata/detect_person/GetListVideoInCameraId/")
async def GetListVideoInCamera(idcam:int,limit: Optional[float] = 10):
    listvideo=GetListVideoInCameraId(idcam,limit)
    return listvideo
UPLOAD_FOLDER= "static"
# @app.get("/getlistCamid")
# def get_list_camid():
#     listcamid=getlistcamid()
#     return listcamid
import os
from PIL import Image
@app.get('/')
def form_post(request: Request):
    result = 'Type a number'
    return templates.TemplateResponse('index.html', context={'request': request})
@app.get('/sysadmin/mlbigdata/detect_person/')
def form_post(request: Request):
    result = 'Type a number'
    return templates.TemplateResponse('index.html', context={'request': request})
@app.get('/sysadmin/mlbigdata/detect_person/file')
def form_post(request: Request):
    result = 'Type a number'
    return templates.TemplateResponse('index.html', context={'request': request})
@app.post("/file")
async def home_page(request: Request,file: UploadFile = File(...) ):
#     try:
#     # Lấy file gửi lên
        image = await file.read()
        if image:
            # Lưu file
            # print(image)
            # print(app.config['UPLOAD_FOLDER'])
            # source1 = os.path.join(UPLOAD_FOLDER,"real"+image.filename)
            # source=os.path.join(UPLOAD_FOLDER,image.filename)
            image_predict = get_predict(model,image)
            path="static/"+file.filename[:-4]+"predict.jpg"
            print(file.filename[:-4])
            im1 = image_predict.save(path)
            path=file.filename[:-4]+"predict.jpg"
            print("image_predict",image_predict)
            # im1 = Image.open(r"/media/quang/New Volume/API/static/0a1aef5326b7b24378c6692f7a454e52.jpg") 
            bytes_io = io.BytesIO()
            image_predict.save(bytes_io, format="PNG")
            result=bytes_io.getvalue()
            #image.save(bytes_io.getvalue(), format='png')
            # print(bytes_io.getvalue())
            # return  templates.TemplateResponse('index.html', context={'request': request, 'result':" bytes_io.getvalue()",'user_image': file.filename   })
            # return Response(bytes_io.getvalue(), media_type="image/png")
            # Trả về kết quả
            return templates.TemplateResponse('index.html', context={'request': request, 'result': path,'path123' : path })
#             return render_template("index.html",real_image = source1,user_image = image.filename, rand= random.random(), msg="Tải file lên thành công", extra=Markup(extra))
 
        else:
            # Nếu không có file thì yêu cầu tải file
            templates.TemplateResponse('index.html', context={'request': request})
            # return render_template('index.html', msg='Hãy chọn file để tải lên')

#     except Exception as ex:
#     # Nếu lỗi thì thông báo
#         print(ex)
#         return templates.TemplateResponse('index.html', context={'request': request})

@app.post("/sysadmin/mlbigdata/detect_person/file")
async def home_page(request: Request,file: UploadFile = File(...) ):
    try:
    # Lấy file gửi lên
        image = await file.read()
        if image:
            # Lưu file
            # print(image)
            # print(app.config['UPLOAD_FOLDER'])
            # source1 = os.path.join(UPLOAD_FOLDER,"real"+image.filename)
            # source=os.path.join(UPLOAD_FOLDER,image.filename)
            image_predict = get_predict(model,image)
            path="static/"+file.filename[:-4]+"predict.jpg"
            print(file.filename[:-4])
            im1 = image_predict.save(path)
            path=file.filename[:-4]+"predict.jpg"
            print("image_predict",image_predict)
            # im1 = Image.open(r"/media/quang/New Volume/API/static/0a1aef5326b7b24378c6692f7a454e52.jpg") 
            bytes_io = io.BytesIO()
            image_predict.save(bytes_io, format="PNG")
            result=bytes_io.getvalue()
            #image.save(bytes_io.getvalue(), format='png')
            # print(bytes_io.getvalue())
            # return  templates.TemplateResponse('index.html', context={'request': request, 'result':" bytes_io.getvalue()",'user_image': file.filename   })
            # return Response(bytes_io.getvalue(), media_type="image/png")
            # Trả về kết quả
            return templates.TemplateResponse('index.html', context={'request': request, 'result': path,'path123' : path, })
            return render_template("index.html",real_image = source1,user_image = image.filename, rand= random.random(), msg="Tải file lên thành công", extra=Markup(extra))
 
        else:
            # Nếu không có file thì yêu cầu tải file
            templates.TemplateResponse('index.html', context={'request': request})
            # return render_template('index.html', msg='Hãy chọn file để tải lên')

    except Exception as ex:
    # Nếu lỗi thì thông báo
        print(ex)
        return templates.TemplateResponse('index.html', context={'request': request})
@app.post("/file1")
def getperson(request: Request, text: str = Form(...)):
    try:
        image_url=text
        img_data = requests.get(image_url).content
        image_predict = get_predict(model,img_data)
        path="static/"+"URLpredict.jpg"
        # print(file.filename[:-4])
        image_predict.save(path)
        path="URLpredict.jpg"
        bytes_io = io.BytesIO()
        image_predict.save(bytes_io, format="PNG")
        result=bytes_io.getvalue()
        return templates.TemplateResponse('index.html', context={'request': request, 'result': path,'path123' : path, })
    except:
        return "loi"
@app.post("/sysadmin/mlbigdata/detect_person/file1")
def getperson(request: Request, text: str = Form(...)):
    try:
        image_url=text
        img_data = requests.get(image_url).content
        image_predict = get_predict(model,img_data)
        path="static/"+"URLpredict.jpg"
        # print(file.filename[:-4])
        image_predict.save(path)
        path="URLpredict.jpg"
        bytes_io = io.BytesIO()
        image_predict.save(bytes_io, format="PNG")
        result=bytes_io.getvalue()
        return templates.TemplateResponse('index.html', context={'request': request, 'result': path,'path123' : path, })
    except:
        return "loi"
@app.post("/getlistCamid")
async def get_list_camid(limit: Optional[int] = 10):
    listcamid=getlistcamid(limit)
    return listcamid
@app.get("/getlistCamEvevator/")
def getlistCamEvevator123():
    listcamid=getlistCamEvevator()
    return listcamid
@app.get("/getlistResult/")
def getlistCamEvevator123():
    listresult=showFile()
    return listresult
# @app.get("/ab")
# def getfile():
#     source="https://cdn.sohatv.vn/KO2G4Q4mOuCYpLu3/encoder/10080468/2021/06/29/10080468_2021-06-29-18-14-58/337a54767a7233703637564336316a6d798448b79e01fa8f1d6a59555ea0002b7509836247d2ee13871d24f3532de549eb56cb344a7edd68f1fc6e80dc0e1ea8abc950909922870c55fdda587bb25e87a6bec88099f1177cd47fce5ad7778eb0c3d6fd969de4e9942ea485f01e2532788c72f78cbea3a1bbbdfa12e4b870ab4e3b75e72196e118c96ca6926fd95877d8c8ada9fd375a22146b6e46dbf90ffa92b891d8bdca6c113f71363f437d4925c77b013095c3b8c35f4621dd1afa24aa86801e447d54c07f4f7016eac5228c1796/10080468_2021-06-29-18-14-58.mp4"
#     list_string=detect(source)
#     print("list:",list_string)
#     # print(list_string)
#     return  list_string
@app.post("/linkvideo/")
async def processvideo(item: str):
    # source="https://cdn.sohatv.vn/KO2G4Q4mOuCYpLu3/encoder/10080468/2021/06/29/10080468_2021-06-29-18-14-58/337a54767a7233703637564336316a6d798448b79e01fa8f1d6a59555ea0002b7509836247d2ee13871d24f3532de549eb56cb344a7edd68f1fc6e80dc0e1ea8abc950909922870c55fdda587bb25e87a6bec88099f1177cd47fce5ad7778eb0c3d6fd969de4e9942ea485f01e2532788c72f78cbea3a1bbbdfa12e4b870ab4e3b75e72196e118c96ca6926fd95877d8c8ada9fd375a22146b6e46dbf90ffa92b891d8bdca6c113f71363f437d4925c77b013095c3b8c35f4621dd1afa24aa86801e447d54c07f4f7016eac5228c1796/10080468_2021-06-29-18-14-58.mp4"
    list_string=detect(item,model,device)
    print("list:",list_string)
    # print(list_string)
    return  list_string
# @app.post("/files/")
# async def create_file(file: bytes = File(...)):
#     return {"file_size": len(file)}

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile = File(...)):
#     return {"filename": file.filename}
    
# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

# @app.post("/create_file/")
# async def image(image: UploadFile = File(...)):
#     print(image.file)
#     # print('../'+os.path.isdir(os.getcwd()+"images"),"*************")
#     try:
#         os.mkdir("images")
#         print(os.getcwd())
#     except Exception as e:
#         print(e) 
#     file_name = os.getcwd()+"/images/"+image.filename.replace(" ", "-")
#     with open(file_name,'wb+') as f:
#         f.write(image.file.read())
#         f.close()
#     img=cv2.imread(file_name)
#     image_predict = get_predict(model,face_detector,img)
#     cv2.imwrite("1.png",img)
#     return {"filename": file}
@app.post("/sysadmin/mlbigdata/detect_person/predictimage")
async def get_predict_map(file: UploadFile = File(...)):
    """Get segmentation maps from image file"""
    image_predict = get_predict(model,await file.read())
   
    
    bytes_io = io.BytesIO()
    image_predict.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")
@app.post("/predictimage")
async def get_predict_map(file: UploadFile = File(...)):
    """Get segmentation maps from image file"""
    image_predict = get_predict(model,await file.read())
   
    
    bytes_io = io.BytesIO()
    image_predict.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")
if __name__=="__main__":
    uvicorn.run(app, port = 4001, host = "172.18.5.16")
