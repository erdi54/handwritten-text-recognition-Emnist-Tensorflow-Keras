from keras.models import load_model
from keras.models import model_from_json
from modelTrain import ModelTrain
import os

image_path = "./image/"
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('model.h5')

model = loaded_model
print('Model successfully loaded')

predict_test = ModelTrain()
letters = predict_test.letters_extract(image_path)
img_to_str = predict_test.img_to_str(model, image_path, letters)

print(img_to_str)
data = img_to_str.split()
keside_yeri = data[0]
tarih = data[1:4]
isim = data[4:6]
tutar_yazi = data[6:10]
tutar = data[10]
# cvs file  , creating a dataset
def openFile(address=r"data.csv"):

    if os.path.exists(address):
        kip = "r+"
    else:
        kip = "w+"
    return open(address,kip)

def dataSave():
    File = openFile()
    registry = "{},{},{},{},{}\n".format(keside_yeri, isim, tarih, tutar, tutar_yazi)
    temp = File.readlines()
    temp.append(registry)
    File.seek(0)
    File.truncate()
    File.writelines(temp)
    File.close()

dataSave()









