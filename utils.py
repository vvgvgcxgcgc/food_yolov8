import os
import zipfile
extracted_folder = '/workdir/radish/nguyenqh/food_yolov8/Vietnamese_Food_data/data'
zip_file = '/workdir/radish/nguyenqh/food_yolov8/data.zip'
print(1)

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)