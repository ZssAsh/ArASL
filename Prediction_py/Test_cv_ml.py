import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import arabic_reshaper
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
from keras.preprocessing import image
from bidi.algorithm import get_display
from moviepy.video.io.bindings import mplfig_to_npimage
import imutils

# 1280 x 720
# VID_WIDTH = 300  # 640 , 1280 x 720
# VID_HEIGHT = 240  # 480
VID_WIDTH = 640
VID_HEIGHT = 480
ROI_W_Percent = 0.39
ROI_H_Percent = 0.52
ROI = [int(VID_WIDTH - (VID_WIDTH * ROI_W_Percent)), int(VID_HEIGHT - (VID_HEIGHT * ROI_H_Percent)),
       int(VID_WIDTH - 10), int(VID_HEIGHT - 10)]

df_Classes = pd.read_excel('ClassLabels.xlsx')
ROI_BORDER_COLOR = (0, 255, 0, 50)
LABEL_COLOR = (0, 255, 0, 0)

lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)
# Mask_Fill = [255, 255, 255] # White Color
Mask_Fill = [0, 0, 0]  # Black Color

showPropChart = False
ShowContours = False
InvertMaskBG = False
FlipImage = True
ShowHelp = True
ContoursthreshL = 45
ContoursthreshH = 255

fig, ax = plt.subplots(figsize=(14, 2), facecolor='w')


# get Class label function
def get_classLabel(class_code, lang='En'):
    if lang == 'En':
        return df_Classes.loc[df_Classes['ClassId'] == class_code, 'Class'].values[0]
    elif lang == 'Ar':
        text_to_be_reshaped = df_Classes.loc[df_Classes['ClassId'] == class_code, 'ClassAr'].values[0]
        reshaped_text = arabic_reshaper.reshape(text_to_be_reshaped)
        return get_display(reshaped_text)


# Resize ROI (Region of Interest) to 64x62 and convert color to Grayscale
def thumbnail(frm):
    crop_img = frm[-(ROI[1] - 1):-11, -(ROI[1] - 1):-11]
    Image64 = cv2.resize(crop_img, (64, 64))
    Image64 = cv2.cvtColor(Image64, cv2.COLOR_BGR2GRAY)
    return Image64


# Create Mask From ROI
def mask_image(frm):
    m = cv2.inRange(hsv, lower_skin, upper_skin)
    # m = cv2.dilate(m, kernel, iterations=5)
    m = cv2.GaussianBlur(m, (3, 3), 50)

    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    # m = cv2.Canny(m, 50, 50)
    m = cv2.GaussianBlur(m, (3, 3), 50)
    # m = cv2.Canny(m, 50, 50)
    return m


def displayBarChart():
    plt.cla()
    h, w, _ = frame.shape
    index = np.arange(3)
    labels = df_Classes['ClassAr']

    # display only to 3

    PredClasses = model.predict(img64Array).reshape(32)
    tempTop3Index = np.argpartition(-PredClasses, 3)
    Top3Index = tempTop3Index[:3]

    tempTop3Values = np.partition(-PredClasses, 3)
    Top3Values = -tempTop3Values[:3]

    for lbl in labels:
        lbl = get_display(arabic_reshaper.reshape(lbl))

    # fig, ax = plt.subplots(figsize=(3, 2), facecolor='w')
    ax.bar(index, Top3Values)
    ax.set_ylabel('Score')
    ax.set_title('Prediction Prop')
    ax.set_xticks(index)
    ax.set_xticklabels(labels)
    plt.tight_layout()

    graphRGB = mplfig_to_npimage(fig)
    gh, gw, _ = graphRGB.shape

    # frame[:gh, w - gw:, :] = mplfig_to_npimage(fig)
    return mplfig_to_npimage(fig)


def displayBarChartFull():
    plt.cla()

    PredClasses = model.predict(img64Array).reshape(32)

    index = np.arange(32)
    labels = df_Classes['ClassAr']

    ax.bar(index, PredClasses, color='green')  # will display blue since cv2 default color order is BRG
    ax.set_ylabel('Score')
    ax.set_title('Prediction Prop')
    ax.set_xticks(index)
    ax.set_xticklabels(labels)

    # plt.tight_layout()

    graphBGR = mplfig_to_npimage(fig)
    # frame[:gh, w - gw:, :] = mplfig_to_npimage(fig)
    # return mplfig_to_npimage(fig)
    return graphBGR


# testing, will be removed later
def Find_contours(frm, frm2):
    image = frm
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    # thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(gray, ContoursthreshL, ContoursthreshH, cv2.THRESH_BINARY)[1]

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # determine the most extreme points along the contour
    # extLeft = tuple(c[c[:, :, 0].argmin()][0])
    # extRight = tuple(c[c[:, :, 0].argmax()][0])
    # extTop = tuple(c[c[:, :, 1].argmin()][0])
    # extBot = tuple(c[c[:, :, 1].argmax()][0])

    cv2.drawContours(frm2, [c], -1, (0, 0, 255), 2)
    # cv2.circle(frm2, extLeft, 8, (0, 255, 255), -1)
    # cv2.circle(frm2, extRight, 8, (0, 255, 255), -1)
    # cv2.circle(frm2, extTop, 8, (0, 255, 255), -1)
    # cv2.circle(frm2, extBot, 8, (0, 255, 255), -1)

    return frm2


model = tf.keras.models.load_model('model_255.h5')
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

camera = cv2.VideoCapture(1)

camera.set(3, VID_WIDTH)
camera.set(4, VID_HEIGHT)

w = camera.get(3)  # cv2.CAP_PROP_FRAME_WIDTH
h = camera.get(4)  # cv2.CAP_PROP_FRAME_HEIGHT

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# fgbg = cv2.createBackgroundSubtractorMOG2()

while camera.isOpened():
    _, frame = camera.read()

    frame = cv2.flip(frame, 1)  # flip the frame horizontally

    roi = frame[ROI[1]:ROI[3], ROI[0]:ROI[2]]

    # Create Mask
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = mask_image(hsv)
    Masked = cv2.bitwise_and(hsv, hsv, mask=mask)
    Masked = cv2.cvtColor(Masked, cv2.COLOR_HSV2BGR)
    indices = np.where(Masked == 0)
    Masked[indices[0], indices[1], :] = Mask_Fill

    # Prepare ROI for Model Prediction
    img64 = thumbnail(Masked)
    if FlipImage:
        img64 = cv2.flip(img64, 1)  # flip the frame horizontally
    img64 = image.img_to_array(img64)
    img64 = img64 / 255
    img64Array = np.array([img64])

    # Predict Classes
    classId = model.predict_classes(img64Array)[0]
    classProb = model.predict(img64Array)[0][classId]
    # classId = np.argmax(classProb, axis=1)

    # print("{} | {}".format(classId, classProb))

    # Prepare Display of Arabic Class Label
    font = ImageFont.truetype("Arial.ttf", 24, encoding="unic")
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    txt = "{}%".format(int(classProb * 100), "%")
    draw.text((ROI[0], ROI[1] - 30), get_classLabel(classId, 'Ar'), font=font, fill=LABEL_COLOR)
    draw.text((ROI[0] + 50, ROI[1] - 30), txt, font=font, fill=LABEL_COLOR)

    # Draw Help mesage
    if ShowHelp:
        txt2 = "Help\n"\
                "h : Toggle this help message\n"\
                "1 : Toggle prediction probability Chart\n"\
                "c : Toggle Contours Display\n"\
                "i : Toggle Mask background (black/white)\n"\
               "\nESC: Quit "
        font2 = ImageFont.truetype("Arial.ttf", 16)
        draw.text((10, 10), txt2, font=font2, fill=LABEL_COLOR)

    frame = np.array(img_pil)

    # Draw ROI Rectangle
    cv2.rectangle(frame, (ROI[0], ROI[1]), (ROI[2], ROI[3]), ROI_BORDER_COLOR, 2)



    if ShowContours:
        aaaaa = Find_contours(Masked, roi)
        frame[ROI[1]:ROI[3], ROI[0]:ROI[2]] = aaaaa

    cv2.imshow('video original', frame)
    cv2.imshow("video original ", img64)

    if showPropChart:
        propChart = displayBarChartFull()
        propChart = cv2.resize(propChart, (700, 100))
        cv2.imshow("Prediction_Chart ", propChart)

    cv2.startWindowThread()

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == 72 or k == 104:  # H or h
        ShowHelp = not ShowHelp
    elif k == 49:  # 1
        showPropChart = not showPropChart
        if not showPropChart:
            cv2.destroyWindow('Prediction_Chart')
            cv2.waitKey(1)
    elif k == 67 or k == 99:  # C or c
        ShowContours = not ShowContours
    elif k == 70 or k == 102:  # F or f
        FlipImage = not FlipImage
    elif k == 73 or k == 105:  # I or i
        if Mask_Fill == [0, 0, 0]:
            Mask_Fill = [255, 255, 255]
            ContoursthreshL = 225
            ContoursthreshH = 255
        else:
            Mask_Fill = [0, 0, 0]
            ContoursthreshL= 45
            ContoursthreshH= 255


