import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread(r'D:\Study\XPlain\testing-images\pepsi+products000518.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(r'C:\Users\rsadiq\Desktop\desktop\New folder\train_images2\p1.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.resize(template, (25, 25), interpolation=cv2.INTER_CUBIC)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.45
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imshow('res.png',img_rgb)
cv2.waitKey(0)