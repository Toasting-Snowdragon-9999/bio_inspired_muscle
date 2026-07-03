"""@brief Convert a source terrain image to grayscale for use as a height field.

Reads ``forrest3.jpg`` as grayscale and writes it out as
``grey_forrest3.png`` so it can be consumed as a Go2 simulation height map.
"""

import cv2

try:
    img = cv2.imread(
        "python/sim/go2/forrest3.jpg",
    cv2.IMREAD_GRAYSCALE
    )
except Exception as e:
    print(f"Error reading image: {e}")
    exit(1)

cv2.imwrite(
    "python/sim/go2/grey_forrest3.png",
    img
)
