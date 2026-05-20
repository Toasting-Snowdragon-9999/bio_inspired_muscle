import cv2

img = cv2.imread(
    "python/sim/go2/checkers_128.png",
    cv2.IMREAD_GRAYSCALE
)

blurred = cv2.GaussianBlur(
    img,
    (15, 15),
    0
)

cv2.imwrite(
    "python/sim/go2/checkers_128_blurred.png",
    blurred
)