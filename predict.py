from ultralytics import YOLO
import numpy as np
import cv2

# Load a model
model = YOLO('/workdir/radish/nguyenqh/food_yolov8/runs/detect/train/weights/best.pt')  # pretrained YOLOv8n model


def predict(input_img,output_path, model = model):# Chỉ cần thay đổi input_img đc đọc = cv2.imread của ảnh cần predict
    # Run batched inference on a list of images
    results = model(source=input_img)  # predict on an image
    bb_lists = [] # mỗi phần tử list này là 1 tuple chứa thông tin về một bounding box
    for result in results:
            result.save(filename=output_path)  # save to disk
            for box in result.boxes:
                left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=np.float128).squeeze()
                width = right - left
                height = bottom - top
                area = width * height
                center = (left + (right-left)/2, top + (bottom-top)/2)
                label_number = int(box.cls) # Số này tương ứng với số thứ tự trong file config.yaml( đánh từ 0);
                label_name = results[0].names[label_number]
                confidence = float(box.conf.cpu())
                t1 = (label_number, center[0],center[1], width, height ) #Tuple biểu diễn anotation của các từng boundingbox dạng <class> <x_center> <y_center> <width> <height>
                t2 = (left, top, right, bottom) #Tuple  biểu diễn các tạo độ của bounding box: left: khoảng từ cạnh trái boundingbox -> lề trái ảnh, right: khoảng từ cạnh phải boundingbox -> lề trái ảnh; top: là khoảng cách từ cạnh trên của boundingbox -> lề trên của ảnh; bottom: là khoảng cách từ cạnh dưới của boundingbox -> lề trên của ảnh 
                t = (t1, t2, area, label_name, confidence) # chứa thông tin về 1 bounding box bao gôm tuple anotation(4 phần tử); tuple biểu diễn các tạo độ của bounding box(4 phần tử); diện tích boudingbox, tên lable( tên món ăn); chỉ số confident.
                bb_lists.append(t)

    return bb_lists

img = cv2.imread('/workdir/radish/nguyenqh/food_yolov8/Data/images/valid/0c93ee4bf2cddcc7_jpg.rf.8703815a75457622063ceda7e7f26296.jpg' )
lst = predict(img,'/workdir/radish/nguyenqh/food_yolov8/predicted_img/res1.jpg')
print(lst)



