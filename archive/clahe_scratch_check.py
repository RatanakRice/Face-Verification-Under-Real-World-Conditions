# Archived: ad hoc one-off check of CLAHE on a single test image, used while
# deciding whether to pursue CLAHE as a lighting correction (see report
# section on the rejected CLAHE correction). Not part of the pipeline; run
# manually with a local test image if re-checking by eye.
import cv2

img = cv2.imread("pic.PNG")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab[:, :, 0] = clahe.apply(lab[:, :, 0])
result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
cv2.imwrite("output.jpg", result)