#ตรวจจับดวงตาและใบหน้า
import cv2
import datetime

# get current date and time
now = datetime.datetime.now()

#อ่านภาพ
img = cv2.imread("dataset_face/19.jpg")


#อ่านไฟล์ xml
face_cascade = cv2.CascadeClassifier("xml/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("xml/haarcascade_eye.xml")

gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#ตรวจจับใบหน้า
face_detect = face_cascade.detectMultiScale(gray_img, 1.3,5)

for (x,y,w,h) in face_detect:
    cv2.rectangle(img,(x,y),(x+w,y+h),(""),thickness=5)
    roi_gray = gray_img[y:y+h, x:x+w]
    face_crop = img[y:y+h, x:x+w]
    re_face =cv2.resize(face_crop,(350,355))


     # Detect eyes
    eyes = eyeCascade.detectMultiScale(roi_gray)
    for i, (ex, ey, ew, eh) in enumerate(eyes):
        cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), thickness=2)
        
        # Crop the eye region
        eye_crop = face_crop[ey:ey+eh, ex:ex+ew]
        
        # Save left eye as even-numbered image and right eye as odd-numbered image
        if ex + ew/2 < w/2: # Left eye
            eye_filename = 'Left_re_eye/eye_crop{}_{}.jpg'.format(now.strftime("%Y%m%d_%H%M%S"), i*2)
        else: # Right eye
            eye_filename = 'Right_re_eye/eye_crop{}_{}.jpg'.format(now.strftime("%Y%m%d_%H%M%S"), i*2+1)
        cv2.imshow(eye_filename, eye_crop) 
        # cv2.imwrite(eye_filename, eye_crop)

#ดวงตา
    y=80
    x=150
    h=95
    w=180
    eye_crop = re_face[y:y+h, x-120:x+w]
#ตาขวา 
    y=80
    x=10
    h=95
    w=180
    eye_cropR = re_face[y:y+h, x:x+w]

#ตาซ้าย
    y=80
    x=160
    h=95
    w=180
    eye_cropL = re_face[y:y+h, x:x+w]
#จมูก
    y=100
    x=80
    h=155
    w=180
    nose_crop = re_face[y:y+h, x:x+w]
#ปาก
    y=240
    x=80
    h=150
    w=180
    mouth_crop = re_face[y:y+h, x:x+w]

face_filename = 're_face/face_{}.jpg'.format(now.strftime("%Y%m%d_%H%M%S"))
nose_filename = 'nose_crop/nose_{}.jpg'.format(now.strftime("%Y%m%d_%H%M%S"))
mouth_filename = 'mouth_crop/mouth_{}.jpg'.format(now.strftime("%Y%m%d_%H%M%S"))

#แสดงภาพ
# cv2.imshow(face_filename, re_face)
cv2.imshow(nose_filename, nose_crop)
cv2.imshow(mouth_filename, mouth_crop)
# cv2.imwrite(face_filename, re_face)
# cv2.imwrite(nose_filename, nose_crop)
# cv2.imwrite(mouth_filename, mouth_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

