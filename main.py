import datetime
# tempfile ; untuk temporarifle gambar yang diuploud ke firebase
#scipy : menghitung jarak berapa lebar mata & mulut
#imutlis videostream bertujuan untuk untuk menghandle data video stream & face utlis untuk mendeteksi wajah
#threading : untuk multi proses agar proses lebih cepat dengan memfaatkan koor komputer yang tersedia
#cv 2 : library untuk mengambil gambar dari kamera
#numpy : untuk pengolahan data angka
import tempfile
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import firebase_admin
from firebase_admin import credentials, db, storage

cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://project-ngantuk-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': 'project-ngantuk.appspot.com'
})

alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
yawn_start_time = None #variabel yang digunakan untuk mencatat waktu ketika pengemudi mulai menguap.
eye_close_start_time = None
last_uptime = 0


detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #sebagai data set kontur wajah
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat') # sebagai detail wajah


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index webcam pada sistem")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
#untuk mata berkedip nilainya 3
EYE_AR_CONSEC_FRAMES = 30
#Jumlah frame berturut-turut di mana EAR harus berada di bawah ambang batas untuk memicu peringatan kantuk.
YAWN_THRESH = 20
#Ambang batas untuk mendeteksi menguap berdasarkan lebar mulut atau metrik yang terkait.


vs = VideoStream(src=args["webcam"]).start()
time.sleep(2.0)


def send_to_firebase(status, value, frame):
    #fungsi untuk mengirim data ke firebase
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(img_bytes)
        temp_file_name = temp_file.name
    bucket = storage.bucket()
    blob = bucket.blob(f"images/{timestamp}.jpg")
    blob.upload_from_filename(temp_file_name)
    os.remove(temp_file_name)


    blob.make_public()

    ref = db.reference('drowsiness')
    ref.push({
        'status': status,
        'value': value,
        'timestamp': timestamp,
        'img': blob.public_url
    })

def alarm(msg):
    #menyalakan dan menutup alarm
    global alarm_status, alarm_status2, saying

    while alarm_status:
        print('call')
        s = 'espeak "' + msg + '"'
        os.system('espeak "{}"'.format(msg))
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system('espeak "{}"'.format(msg))
        os.system(s)
        saying = False

def eye_aspect_ratio(eye):
    #untuk menentukan buka dan menutup mata di bagian aspect ratio
    #kenapa dimulai dari 0? karena data aray
    #mata akan dideteksi ketika berkedip, jarak pixxel antara mata tidak berubah akan terdeteksi ( Mengantuk / tidak terbuka/ tertutup)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3]) #panjang kelopak mata
    ear = (A + B) / (2.0 * C) # untuk menghitung jarak terdekat dari panjangnya mata kanan dan kiri
    return ear

def final_ear(shape):
    #yg dideteksi pada bagian ini adalah jarak pixel antara kedua mata
    # final ear membandingkan mata kanan dan kiri, jarak mata kanan & kiri
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd] # LeftEYE untuk mengambil gambar 1 mata kiri
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye) # left Ear untuk menghitung jarak mata berkedip / tidak
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    #algoritma untuk mendeteksi mulut
    top_lip = shape[50:54]
    top_lip = np.concatenate((top_lip, shape[61:65]))
    low_lip = shape[56:60]
    low_lip = np.concatenate((low_lip, shape[65:61]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Main loop
while True:
    #berfungsi untuk menghambil fream data stream
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                      #mengambil gambar wajah
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]
        distance = lip_distance(shape)
        #mengambil data mata

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        #fungsi adalah memunculkan gambar bulatan kuning pada mata dan mulut
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            #jarak minimum mata terbuka/tertutup
            if eye_close_start_time is None: #digunakan untuk mencatat waktu saat mata mulai tertutup.
                eye_close_start_time = time.time() #maka inisialisasi dengan waktu saat ini. time.time() mengembalikan waktu saat ini dalam detik
            elif time.time() - eye_close_start_time > 4: #Mengecek apakah mata telah tertutup selama lebih dari 4 detik
                COUNTER += 1 #Jika mata telah tertutup selama lebih dari 4 detik,nilai COUNTER yang digunakan untuk melacak jumlah frame berturut-turut di mana mata tertutup.
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "MATA MENGANTUK!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not alarm_status:
                        alarm_status = True
                        t = Thread(target=alarm, args=('wake up sir',))
                        t.deamon = True
                        t.start()
                        send_to_firebase("Mata Mengantuk !!!", ear, frame)
        else:
            COUNTER = 0 #variabel yang digunakan untuk menghitung jumlah frame berturut-turut di mana mata terdeteksi tertutup
            alarm_status = False #berarti alarm tidak akan berbunyi karena mata tidak lagi terdeteksi tertutup dalam waktu yang lama.
            eye_close_start_time = None #digunakan untuk mencatat waktu saat mata mulai tertutup. Mengatur ulang variabel ini ke None berarti waktu mulai untuk mata tertutup dihapus karena mata tidak lagi tertutup.

        if distance > YAWN_THRESH:
            if yawn_start_time is None:  # Jika mulut menguap, tetapi waktu mulai belum diatur
                yawn_start_time = time.time()  # Setel waktu mulai
            elif time.time() - yawn_start_time > 2:  # Hanya deteksi setelah mulut menguap selama 7 detik
                cv2.putText(frame, "MULUT MENGUAP !!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not alarm_status2 and not saying:
                    alarm_status2 = True
                    t = Thread(target=alarm, args=('take some fresh air sir',))
                    t.deamon = True
                    t.start()
                    send_to_firebase("Mulut Menguap !!!", distance, frame)
        else:
            alarm_status2 = False
            yawn_start_time = None

        cv2.putText(frame, "MATA: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # berfungsi untuk menampilkan kontur mata
        cv2.putText(frame, "MULUT: {:.2f}".format(distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    now = time.time()
    if (now - last_uptime) > 10:
        db.reference('last_uptime').set(time.strftime("%Y-%m-%d %H:%M:%S"))
        last_uptime = now

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()