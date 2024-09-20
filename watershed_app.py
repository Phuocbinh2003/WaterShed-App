import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt


def apply_watershed(img):

    img_bgr = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

    blurred = cv2.medianBlur(img_bgr, ksize=3)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Binarization
    ret, image_thres = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tạo mask cho Watershed
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(
        image_thres, cv2.MORPH_OPEN, kernel=kernel, iterations=2)

    # Distance Transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Show Sure Foreground
    ret, sure_foreground = cv2.threshold(
        src=dist_transform, thresh=0.08*np.max(dist_transform), maxval=255, type=0)

    # Show Sure BackGround
    sure_background = cv2.dilate(
        src=opening, kernel=kernel, iterations=2)  # int

    # change its format to int
    sure_foreground = np.uint8(sure_foreground)

    # Show Unknow
    unknown = cv2.subtract(sure_background, sure_foreground)

    # Gắn nhãn markers
    ret, marker = cv2.connectedComponents(sure_foreground)
    marker = marker + 1
    marker[unknown == 255] = 0

    # Áp dụng Watershed transform
    marker = cv2.watershed(image=img_bgr, markers=marker)

    # Segmented Objects
    contour, hierarchy = cv2.findContours(
        image=marker.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    image_vis = img_bgr.copy()

    for i in range(len(contour)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(image=image_vis, contours=contour,
                             contourIdx=i, color=(0, 0, 255), thickness=1)

    # Trả về ảnh kết quả
    return img_bgr, blurred, image_thres, dist_transform, sure_foreground, sure_background, unknown, marker, marker, image_vis


# Xây dựng ứng dụng
st.title('✨ License Plate Detection App ')

st.sidebar.write("## 📷 Upload Image")
uploaded_file = st.sidebar.file_uploader("", type="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Thực hiện nhận diện biển số bằng Watershed
    if st.button('Detect License Plate'):
        (img_bgr, blurred, image_thres, dist_transform, sure_foreground,
         sure_background, unknown, marker, watershed_image, image_vis) = apply_watershed(img)

        # Tạo lưới subplot
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

        # Hiển thị các kết quả trung gian
        axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 1].imshow(blurred, cmap='gray')
        axes[0, 1].set_title('Blurred Image')
        axes[0, 2].imshow(image_thres, cmap='gray')
        axes[0, 2].set_title('Threshold Image')
        axes[1, 0].imshow(dist_transform, cmap='gray')
        axes[1, 0].set_title('Distance Transform')
        axes[1, 1].imshow(sure_foreground, cmap='gray')
        axes[1, 1].set_title('Sure Foreground')
        axes[1, 2].imshow(sure_background, cmap='gray')
        axes[1, 2].set_title('Sure Background')
        axes[2, 0].imshow(unknown, cmap='gray')
        axes[2, 0].set_title('Unknown')
        axes[2, 1].imshow(marker, cmap='gray')
        axes[2, 1].set_title('Marker')
        axes[2, 2].imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
        axes[2, 2].set_title('Watershed Image')

        # Ẩn các trục
        for ax in axes.flatten():
            ax.axis('off')

        # Hiển thị subplot
        st.pyplot(fig)
