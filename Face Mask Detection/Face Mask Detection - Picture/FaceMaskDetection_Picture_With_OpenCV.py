# Kullanım
# python FaceMaskDetection_Picture_With_OpenCV.py --image examples/example_01.png

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def FaceMaskDetection_Picture_With_OpenCV_HaarCascade(image_file, model_file, cascade_classifier):
    faceCascade = cv2.CascadeClassifier(cascade_classifier)

    model = load_model(model_file)

    image = plt.imread(image_file)

    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("> {0} adet yüz bulundu!".format(len(faces)))

    # Bulunan yüzleri dörgen ile çerçeveleyelim
    for (x, y, w, h) in faces:
        startX = x
        startY = y
        endX = x + w
        endY = y + h

        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # Elde edilen yüzü, maske varlığını test etmek için modelimize yolluyoruz
        (withoutMask, mask) = model.predict(face)[0]

        # Modelimizin değerlendirmesi sonucunda elde ettiğimiz sınıf etiketlerini
        #   kullanarak, sonuç ve sonuç karesinin rengini belirliyoruz
        label = "Maske Var" if mask > withoutMask else "Maske Yok"
        color = (0, 255, 0) if label == "Maske Var" else (0, 0, 255)

        # Oluşturulan etikete modelin döndürdüğü olasılığı da ekliyoruz
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Çıktı çerçevesine etiketi ve yüzü çevreleyen dörtgeni çiziyoruz
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

    # İşlenen görüntüyü sonuç olarak ekrana ve diske yazdırıyoruz
    cv2.putText(image, "[With OpenCV CascadeClassifier] *** NOT OK ***", (5,20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1)
    cv2.imwrite("DetectMask_OpenCV_HaarCascade.jpg", image, )
    cv2.imshow("Output", image)
    cv2.waitKey(0)

# Program satırından başlatıldığı durumlar için gerekecek argüman düzenleyici
#   ile parametreleri ve varsayılan değerlerini ortama tanıtıyoruz
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    default="test_images/0072.jpg",
    help="İmaj dosyasının diskteki yolu")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="Eğitilmiş yüz maskesi detektörünün diskteki yolu")
ap.add_argument("-c", "--classifier", type=str,
	default=r'face_detector\haarcascade_frontalface_default.xml',
	help="OpenCV yüz tanıma sınıflandırıcısı <Haar Cascade Classifier>ın diskteki yolu")
args = vars(ap.parse_args())

FaceMaskDetection_Picture_With_OpenCV_HaarCascade(args["image"], args["model"], args["classifier"])
