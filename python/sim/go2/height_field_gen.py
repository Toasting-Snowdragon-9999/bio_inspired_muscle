"""@brief Generate a smoothed grayscale height field from a checkerboard image.

Reads ``checkers_128.png`` as grayscale, applies a Gaussian blur to soften
the hard edges, and writes the result to ``checkers_128_blurred.png`` for use
as a Go2 simulation height map.
"""

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