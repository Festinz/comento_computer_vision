import cv2
import numpy as np

# 이미지로드
image = cv2.imread('sample.jpg')  # 분석할 이미지 파일
# BGR에서HSV 색상공간으로변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# 빨간색범위지정(두개의범위를설정해야함)
lower_red1 = np.array([0, 120, 70])  
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])  
upper_red2 = np.array([180, 255, 255])
# 마스크생성
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2  # 두개의마스크를합침
# 원본이미지에서빨간색부분만추출
result = cv2.bitwise_and(image, image, mask=mask)
# 결과이미지출력
cv2.imshow('Original', image)
cv2.imshow('Red Filtered', result)
cv2.waitKey(0)
cv2.destroyAllWindows()