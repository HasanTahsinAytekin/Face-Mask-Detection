import argparse
import time

import cv2
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Kameradan elde edilen görüntünün maske tahminini yapan fonksiyon
def detect_and_predict_mask(frame, faceNet, maskNet, face_confidence_probability):
    # Çerçevenin boyutlarını aldıktan sonra yüz tanımlama için bir blob oluştur
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Oluşturulan blob'u, yüz araştırması için nöral ağa yolla
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # İlgili yüzler, yerleri ve olasılıkları için yeni dizinler oluşturalım
    faces = []
    locs = []
    preds = []

    # Bulunan her bir yüz için işlemleri tekrarlayalım
    for i in range(0, detections.shape[2]):
        # Bulunan yüzün güven (confidence) olasılığını belirleyelim
        confidence = detections[0, 0, i, 2]

        # Yüz verisi için elde ettiğimiz güven oranı, daha önce belirlediğimiz kabul
        #   edilebilir en düşük güven aralığından büyükse işleme devam edelim
        if confidence > face_confidence_probability:
            # cYüz nesnesinin koordinatlarını tespit edelim
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Çevreleyen kutu boyutları dışarı taşmasın
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # İşlemek için yüz üzerinde işlemlerimizi tamamlayalım
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Yüz ve yüzü çevreleyecek kutunun bilgilerini ilgili listelere ekleyelim
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # En az bir yüz tespit edilmiş olmalı
    if len(faces) > 0:
        preds = maskNet.predict(faces)

    # Yüz koordinatları ve maske sonucunu ikili olarak bir araya getirip sonuç olarak geri dödürelim
    # locations
    return (locs, preds)

# Bilgisayarın kamerasından elde edilen görüntülerde kişiyi arayan ve
#   bulunacak kişilerde maske araştırması yapacak fonksiyon.
def FaceMaskDetection_Camera(model_file, prototxt_file, weights_file, face_confidence_probability):
    # OpenCV DNN yüz detektörü modelini yükleyelim
    print("> Yüz dedektörü modeli yükleniyor ...")
    faceNet = cv2.dnn.readNet(prototxt_file, weights_file)

    # Bizim oluşturduğumuz yüz maskesi detektör modelini yükleyelim
    print("> Yüz maskesi detektör modeli yükleniyor ...")
    maskNet = load_model(model_file)

    # Video akışını başlatalım ve kamera sensörüne hazırlanması için zaman tanıyalım
    print("> Video akışı başlatılıyor...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    print("> Video akışını sonlandırmak için <Esc> tuşuna basın...")

    # Video akışından devamlı çerçeve resimler alalım
    while True:
        # Video akışından bir çerçeve alalım
        frame = vs.read()

        # Çerçevedeki yüzleri bulalım ve maske takıp takmadıklarını kontrol edelim
        try:
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, face_confidence_probability)
        except cv2.error:
            pass

        # Bulunan her yüzün ilgili koordinatlarını kullanarak sonuç elde edelim
        for (box, pred) in zip(locs, preds):
            # Çerçeve koordinatları ve maske olasılıklarını veriden çıkartalım
            (startX, startY, endX, endY) = box
            (withoutMask, mask) = pred

            # Modelimizin değerlendirmesi sonucunda elde ettiğimiz sınıf etiketlerini
            # 	kullanarak, sonuç ve sonuç karesinin rengini belirliyoruz
            label = "Maske VAR" if mask > withoutMask else "Maske YOK"
            color = (0, 255, 0) if label == "Maske VAR" else (0, 0, 255)

            # Oluşturulan etikete modelin döndürdüğü olasılığı da ekliyoruz
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Çıktı çerçevesine etiketi ve yüzü çevreleyen dörtgeni çiziyoruz
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

        # Üzerinde gerçekleştirdiğimi işlemler tamamlandığı için, sonuç çerçeveyi ekrana yaz
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Eğer 'q' veya Esc tuşuna basılırsa, döngüyü sonlandır
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
    vs.stop()

# Program satırından başlatıldığı durumlar için gerekecek argüman düzenleyici
#   ile parametreleri ve varsayılan değerlerini ortama tanıtıyoruz
ap = argparse.ArgumentParser()
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

FaceMaskDetection_Camera(args["model"], args["prototxt"], args["weights"], args["confidence"])
