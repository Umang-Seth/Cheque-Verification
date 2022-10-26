import cv2
import numpy as np

def empty(a):
    pass

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("Threshold2", "Trackbars", 10, 255, empty)

def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1, Threshold2
    return src

def preProcessing(img):
    #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(img,(3,3),1)
    imgCanny = cv2.Canny(imgBlur, 0, 1)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (0, 0, 255), 10)
    #print(biggest)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    #print("diff",diff)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print(myPointsNew)
    return myPointsNew


def getWarp(img, biggest):
    if biggest.size != 0:
        biggest = reorder(biggest)
    else:
        return 0
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [Width, 0], [0, Height], [Width, Height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (Width, Height))
    return imgOutput

def sign_extraction(img):
    imgCropped = img[int(Height/2):Height-80,int(Width-Width/3):Width]
    return imgCropped

def payee_extraction(img):
    imgCropped = img[53:80,65:500]
    return imgCropped

def rs_extraction(img):
    imgCropped = img[75:105,95:Width]
    return imgCropped

def amount_extraction(img):
    imgCropped = img[100:133,495:Width]
    return imgCropped

def date_extraction(img):
    imgCropped = img[18:35,490:620]
    return imgCropped

def accno_extraction(img):
    imgCropped = img[140:165,55:255]
    return imgCropped

initializeTrackbars()
count = 0

while True:
    #success, img = cap.read()
    img = cv2.imread("Photo Cheque/sampleCheque.png",0)
    sign = cv2.imread("Photo Cheque/sign.png",0)
    image = image_resize(img, width=640, height=286)#(1280,567)(1280,582)(1280,572)
    Width = image.shape[1]
    Height = image.shape[0]
    #print("width",Width,"height",Height)
    #img = cv2.resize(img, (Width, Height))
    imgContour = image.copy()
    imgThres = preProcessing(image)
    Biggest = getContours(imgThres)
    imgWarp = getWarp(image, Biggest)
    cv2.imshow("Show",sign)
    #cv2.imshow("Original",image)
    #cv2.imshow("Original", imgContour)
    #cv2.imshow("Canny", imgThres)

    if Biggest.size != 0:
        imgCropped = imgWarp
        cv2.imshow("Output", imgCropped)
        #imgWarpGray = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)
        imgBin = cv2.threshold(imgCropped, 140, 255, cv2.THRESH_BINARY)[1]
        imgSign = sign_extraction(imgCropped)
        imgPayee = payee_extraction(imgBin)
        imgRs = rs_extraction(imgBin)
        imgDate = date_extraction(imgBin)
        imgAmount = amount_extraction(imgBin)
        imgAccno = accno_extraction(imgBin)
        cv2.imshow("Final", imgBin)
        cv2.imshow("Sign", imgSign)
        cv2.imshow("Payee", imgPayee)
        cv2.imshow("Rs", imgRs)
        cv2.imshow("Date",imgDate)
        cv2.imshow("Amount",imgAmount)
        cv2.imshow("Acc No.",imgAccno)

    else:
        cv2.imshow("Blank", np.zeros((Height, Width, 3), np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(count)
        cv2.imwrite("ScannedImage" + str(count) + ".jpg", imgBin)
        cv2.imwrite("OriginalImage" + str(count) + ".jpg", imgCropped)
        cv2.waitKey(300)
        count += 1