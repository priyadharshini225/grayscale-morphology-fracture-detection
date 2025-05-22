# 🦴 Bone Fracture Detection using Grayscale Morphology (OpenCV)

This project demonstrates a **basic-level implementation** of real-time bone fracture detection using classical image processing techniques like **grayscale morphological operations** and **edge detection** with OpenCV.

It is designed as a simple prototype to process X-ray images, detect possible bone fractures, and visualize the results using bounding boxes.

---

## 📌 Features

- Read and preprocess grayscale X-ray images
- Enhance contrast using histogram equalization
- Detect fracture-like patterns using:
  - Morphological gradient
  - Canny edge detection
  - Contour analysis
- Visual output with suspected fracture regions highlighted

---

## 🧰 Requirements


- Python 3.x
- OpenCV
- Matplotlib

### PROGRAM
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("fractured_b.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (512, 512))  # Resize for uniformity
image[0:50, 0:150] = 0  
image[0:50, -150:] = 0
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(image)


# Step 2: Enhance contrast using Histogram Equalization
equalized = cv2.equalizeHist(image)


blurred = cv2.GaussianBlur(equalized, (3, 3), 0)



# Step 4: Apply Morphological Gradient (dilation - erosion)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morph_gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)



# Step 5: Apply Canny edge detection
edges = cv2.Canny(morph_gradient, threshold1=50, threshold2=150)




# Step 6: Draw contours around suspected fracture areas
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if 150 < area < 2500:  # Heuristic: filter small areas
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)


adaptive = cv2.adaptiveThreshold(
    morph_gradient, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    11, 2
)
edges = cv2.Canny(adaptive, 30, 100)

plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original X-ray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(morph_gradient, cmap='gray')
plt.title('Morph Gradient')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(output, cmap='gray')
plt.title('Detected Fractures')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### OUTPUT


![image](https://github.com/user-attachments/assets/dff0b11e-465e-466d-97af-96d31d3bacf1)


## RESULT:
Thus the output for the bone fracture detection is done sucessfully.
