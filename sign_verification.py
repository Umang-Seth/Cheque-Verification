import cv2
import numpy as np

def sign_verification(img1,img2):
    orb = cv2.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    return img3


img1 = cv2.imread("Photo Cheque/sign.png",0)
img2 = cv2.imread("Photo Cheque/sampleCheque.png",0)
img4 = cv2.imread("Photo Cheque/umang.jpg")

Height = img2.shape[0]
Width = img2.shape[1]
img2Croppped = img2[int(Height/2):Height-100,int(Width-Width/3):Width]

print(sign_verification(img1,img2Croppped))
print(sign_verification(img4,img2Croppped))
#img3 = cv2.drawMatchesKnn(img1,kp1,img2Croppped,kp2,good,None,flags=2)

#imgKp1 = cv2.drawKeypoints(img1,kp1,None)
#imgKp2 = cv2.drawKeypoints(img2Croppped,kp2,None)

#cv2.imshow("kp1",imgKp1)
#cv2.imshow("kp2",imgKp2)
cv2.imshow("img1",img1)
cv2.imshow("img2",img2Croppped)
cv2.imshow("img3",img4)
cv2.imshow("img4",sign_verification(img1,img2Croppped))
cv2.imshow("img5",sign_verification(img4,img2Croppped))
#cv2.imshow("img3",img3)
cv2.waitKey(0)