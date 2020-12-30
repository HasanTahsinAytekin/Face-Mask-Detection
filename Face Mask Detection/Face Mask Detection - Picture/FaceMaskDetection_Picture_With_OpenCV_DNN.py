# Kullanım
# python FaceMaskDetection_Picture_With_OpenCV_DNN.py --image examples/example_01.png

import argparse

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def FaceMaskDetection_Picture_With_OpenCV_DNN(image_file, model_file, prototxt_file, weights_file, face_confidence_probability):
	#
	print("> DNN yüz tanımlama modeli yükleniyor...")
	# OpenCV DNN modelin diskten okuyalım
	net = cv2.dnn.readNet(prototxt_file, weights_file)

	# Maske belirleme amacıyla oluşturduğumuz modeli diskten okuyalım
	print("> Yüz maskesi tanımlama modeli yükleniyor...")
	model = load_model(model_file)

	image = cv2.imread(image_file)

	(h, w) = image.shape[:2]

	# Yüklenen görüntüden blop oluşturalım
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# Oluşturulan blob'u var olan potansiyel yüzleri algılamak için DNN'e yollayalım
	print("> Yüzler algılanıyor...")
	net.setInput(blob)
	detections = net.forward()

	print("> {0} adet yüz olmaya aday nesne bulundu!".format(detections.shape[2]))

	faceCount = 0
	# Bulunan bütün yüzler için
	for i in range(0, detections.shape[2]):
		# Geri yollanan nesnenin yüz olma olasılığını alalım
		confidence = detections[0, 0, i, 2]

		# Yüzdesi parametre olarak yolladığımız değerden büyük olan yüzleri inceleyelim
		if confidence > face_confidence_probability:
			# Yüz nesnesinin etrafına çizeceğimiz diktörgen için ilgili koordinatlarını belirleyelim
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Belirlediğimiz koordinatlar dışa taşmasın
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Yüzü oluşturduğumuz modele yollamak için hazırlık yapalım
			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# Elde edilen final görüntüyü işlemek için oluşturduğumuz modelimize yollayalım
			(withoutMask, mask) = model.predict(face)[0]

			# Modelimizin değerlendirmesi sonucunda elde ettiğimiz sınıf etiketlerini
			# 	kullanarak, sonuç ve sonuç karesinin rengini belirliyoruz
			label = "Maske Var" if mask > withoutMask else "Maske Yok"
			color = (0, 255, 0) if label == "Maske Var" else (0, 0, 255)

			# Oluşturulan etikete modelin döndürdüğü olasılığı da ekliyoruz
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# Çıktı çerçevesine etiketi ve yüzü çevreleyen dörtgeni çiziyoruz
			cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
			#
			faceCount = faceCount + 1

	print("> {0} adet yüz belirlendi!".format(faceCount))
	# İşlenen görüntüyü sonuç olarak ekrana ve diske yazdırıyoruz
	cv2.putText(image, "[With OpenCV DNN] *** NOT OK ***", (5,20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 0, 0), 1)
	cv2.imwrite("DetectMask_OpenCV_DNN.jpg", image, )
	cv2.imshow("Output", image)
	cv2.waitKey(0)


# Program satırından başlatıldığı durumlar için gerekecek argüman düzenleyici
#   ile parametreleri ve varsayılan değerlerini ortama tanıtıyoruz
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="İmaj dosyasının diskteki yolu")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="Eğitilmiş yüz maskesi detektörünün diskteki yolu")
ap.add_argument("-p", "--prototxt", type=str,
	default=r'face_detector\deploy.prototxt',
	help="Eğitimli DNN dosyası olan Caffe “deploy” prototxt'in diskteki yolu")
ap.add_argument("-w", "--weights", type=str,
	default=r'face_detector\res10_300x300_ssd_iter_140000.caffemodel',
	help="Daha önce eğitlmiş olan DNN Caffe modelinin ağırlıklarının olduğu dosyanın diskteki yolu")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="Bulunan yüzlerin minimum olasılıkları")
args = vars(ap.parse_args())

FaceMaskDetection_Picture_With_OpenCV_DNN(args["image"], args["model"], args["prototxt"], args["weights"], args["confidence"])
