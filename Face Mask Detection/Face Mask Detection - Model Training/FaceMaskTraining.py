import json
import os
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator


# Yüz maskesi tanıma modeli tanımlama sınıfı
# Bu sınıf ile model oluşturulmakta ve çalışma
#   özellikleri belirlenmektedir.
class FaceMaskModel(tf.keras.Model):
    def __init__(self):
        super(FaceMaskModel, self).__init__()
        # Temel model
        self.AP = AveragePooling2D(pool_size=(7, 7))
        self.F = Flatten(name="flatten")
        self.D1 = Dense(128, activation="relu")
        self.DRO = Dropout(0.5)
        self.D2 = Dense(2, activation="softmax")

    def call(self, base_model):
        # Temel modelin üzerine kurulacak katmanları belirliyoruz
        head_model = base_model.output
        head_model = self.AP(head_model)
        head_model = self.F(head_model)
        head_model = self.D1(head_model)
        head_model = self.DRO(head_model)
        head_model = self.D2(head_model)
        #
        for layer in base_model.layers:
            layer.trainable = False
        #
        # Ana yüz tanıma modelini, temel modelin üzerine yerleştiriyoruz,
        #   böylece eğitim için kullanacağımız modelin tamamı oluşuyor.
        return Model(inputs=base_model.input, outputs=head_model)

# Bu sınıf ile, daha önce tanımlanmış yüz tanıma modeli eğitilmektedir.
class FaceMaskTraining():
    def __init__(self):
        # Varsayılanları oluştur (Class Variables)
        self.image_directory = r'C:\ML\Datasets\FaceMaskDetection\archive2\Medical mask\Medical mask\Medical Mask\images'
        self.annotation_directory = r'C:\ML\Datasets\FaceMaskDetection\archive2\Medical mask\Medical mask\Medical Mask\annotations'
        self.initial_learning_rate = 1e-4
        self.epochs = 30
        self.batch_size = 32
        self.images = []
        self.labels = []

    # Bütün class variable'ları yeniden belirle
    def __init__(self, image_directory, annotation_directory, initial_learning_rate, epochs, batch_size):
        # Varsayılanları Değiştir
        self.image_directory = image_directory
        self.annotation_directory = annotation_directory
        self.initial_learning_rate = initial_learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.images = []
        self.labels = []

    # Sadece image_directory ve annotation_directory class variable'larını yeniden belirle
    @classmethod
    def __init__1(self, image_directory, annotation_directory):
        self.__init__()
        # Varsayılanları Değiştir
        self.image_directory = image_directory
        self.annotation_directory = annotation_directory

    # Sadece initial_learning_rate, epochs ve batch_size class variable'larını yeniden belirle
    @classmethod
    def __init__2(self, initial_learning_rate, epochs, batch_size):
        self.__init__()
        # Varsayılanları Değiştir
        self.initial_learning_rate = initial_learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    # Görüntüleri diskten etiketleri ile birlikte yükleyerek düzenleyen metod
    def load_images(self):
        print("> Görüntüler yükleniyor...")
        for filename in os.listdir(self.image_directory):
            num = filename.split('.')[0]
            print('\r> Yüklenen görüntü: {}'.format(filename), end="")
            if int(num) > 1800:
                class_name = None
                anno = filename + ".json"
                with open(os.path.join(self.annotation_directory, anno)) as json_file:
                    json_data = json.load(json_file)
                    no_anno = json_data["NumOfAnno"]
                    k = 0
                    for i in range(0, no_anno):
                        class_nam = json_data['Annotations'][i]['classname']
                        if class_nam in ['face_with_mask', "gas_mask", "face_shield", "mask_surgical", "mask_colorful"]:
                            class_name = 'face_with_mask'
                            k = i
                            break
                        elif class_nam in ['face_no_mask,"hijab_niqab', 'face_other_covering',
                                           "face_with_mask_incorrect",
                                           "scarf_bandana", "balaclava_ski_mask", "other"]:
                            class_name = 'face_no_mask'
                            k = i
                            break
                        else:
                            continue

                    box = json_data['Annotations'][k]['BoundingBox']
                    (x1, x2, y1, y2) = box
                if class_name is not None:
                    image = cv2.imread(os.path.join(self.image_directory, filename))
                    img = image[x2:y2, x1:y1]
                    img = cv2.resize(img, (224, 224))
                    img = img[..., ::-1].astype(np.float32)
                    img = preprocess_input(img)
                    self.images.append(img)
                    self.labels.append(class_name)

        self.images = np.array(self.images, dtype="float32")
        self.labels = np.array(self.labels)

        print('\r> Yüklenen görüntü sayısı: ' + str(len(self.images)))
        print('> Yüklenen ek açıklama (etiket) sayısı: ' + str(len(self.labels)))
        print('>    face_with_mask: ' + str(np.count_nonzero(self.labels == 'face_with_mask', axis=0)))
        print('>    face_no_mask  : ' + str(np.count_nonzero(self.labels == 'face_no_mask', axis=0)))

    # Daha sonra kullanılacak olan yüz tanıma kategorizasyonu yapacak modelin
    #   oluşturulması, disk üzerinde serileştirilerek saklanması, ve serileştirme
    #   işleminden elde edilecek verilerin raporlanmasını sağlayan metod.
    def build_save_report_model(self):
        print("> Maske tanıma modeli oluşturuluyor ...")
        # Etiketler kategori oluşturmak için kodlanacak
        lb = LabelBinarizer()
        labels_loaded = lb.fit_transform(self.labels)
        labels_loaded = to_categorical(labels_loaded)

        # Verilerin %75'ini eğitim için ve kalan %25'ini de test için kullanabilmek
        #   amacıyla veriler eğitim (train) ve test olarak iki bölüme ayrılmaktadır.
        (trainX, testX, trainY, testY) = train_test_split(self.images, labels_loaded,
                                                          test_size=0.20, stratify=labels_loaded, random_state=42)

        # Gerçek zamanlı veri artırma ile eğitim için kullanılacak toplu tensör görüntü verilerini oluşturun
        aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")

        model = FaceMaskModel()
        # Modeli MobileNetV2 ağı üzerine kurmaktayız.
        model = model(MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))))

        # Oluşturulan modelin özetini yazdıralım
        print(">\n> Oluşturulan Modelin Özeti:")
        print(model.summary())
        print(">")

        # Oluşturulan modeli derleyelim
        print("> Yüz tanıma modeli derleniyor...")
        opt = Adam(lr=self.initial_learning_rate, decay=self.initial_learning_rate / self.epochs)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        # Derlenen modeli eğitelim
        print("> Yüz tanıma modeli eğitiliyor...")
        history = model.fit(
            aug.flow(trainX, trainY, batch_size=self.batch_size),
            steps_per_epoch=len(trainX) // self.batch_size,
            validation_data=(testX, testY),
            validation_steps=len(testX) // self.batch_size,
            epochs=self.epochs)

        # Test setiyle eğitilen modelin tahmin performansına bakalım (keras.engine.training.Model.predict)
        print("\n> Girdi örnekleri için çıktı tahminleri oluşturuluyor...")
        predIdxs = model.predict(testX, batch_size=self.batch_size)

        # Test setindeki her bir görüntü için, en büyük tahmin olasılığına karşılık gelen etiketin dizinini bulmamız gerekiyor.
        predIdxs = np.argmax(predIdxs, axis=1)

        # Elde edilen sınıflandırmayı rapolayalım
        print(classification_report(testY.argmax(axis=1), predIdxs,
                                    target_names=lb.classes_))

        # Oluşturduğumuz ve Eğittiğimiz modeli dizin haline getirerek (serialize) diske kaydedelim
        print("> Maske tanıma modeli diske yazılıyor...")
        model.save("mask_detector_test.model", save_format="h5")

        # Eğitimden kaynaklanan kayıp (loss) ve doğruluk (accuracy) verilerini raporlayalım
        print("> Maske tanıma modeli eğitim istatistik grafikleri oluşturuluyor...")
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), history.history["loss"], label="Eğitim kaybı")
        plt.plot(np.arange(0, self.epochs), history.history["val_loss"], label="Doğrulama kaybı")
        plt.plot(np.arange(0, self.epochs), history.history["accuracy"], label="Eğitim doğruluğu")
        plt.plot(np.arange(0, self.epochs), history.history["val_accuracy"], label="Doğrulama doğruluğu")
        plt.title("Eğitim Kaybı ve Doğruluğu")
        plt.xlabel("Dönem #")
        plt.ylabel("Kayıp/Doğruluk")
        plt.legend(loc="lower left")
        plt.savefig("FaceMaskTrainingStats.png")
        plt.show()

        print("> Maske tanıma modeli oluşturuldu.")


# Program satırından başlatıldığı durumlar için gerekecek argüman düzenleyici
#   ile parametreleri ve varsayılan değerlerini ortama tanıtıyoruz
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imageFolder", type=str,
	default=r'C:\ML\Datasets\FaceMaskDetection\archive2\Medical mask\Medical mask\Medical Mask\images',
	help="Resimlerin olduğu klasörün diskteki yolu")
ap.add_argument("-a", "--annotationFolder", type=str,
	default=r'C:\ML\Datasets\FaceMaskDetection\archive2\Medical mask\Medical mask\Medical Mask\annotations',
	help="Etiketlerin olduğu klasörün diskteki yolu")
args = vars(ap.parse_args())

maskTraining = FaceMaskTraining(args["imageFolder"], args["annotationFolder"], 1e-4, 30, 32)
maskTraining.load_images()
maskTraining.build_save_report_model()
