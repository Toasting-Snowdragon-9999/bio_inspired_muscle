import cv2
import numpy as np

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
SIZE = 512           # Terrain resolution
OCTAVES = 10            # More octaves = more detail
PERSISTENCE = 0.5      # Amplitude falloff
LACUNARITY = 2.0       # Frequency growth
SCALE = 120.0         # Larger = smoother terrain

OUTPUT = "python/sim/go2/huge_perlin_map.png"

# ---------------------------------------------------
# Perlin noise implementation
# ---------------------------------------------------
def fade(t):
    return 6*t**5 - 15*t**4 + 10*t**3

def lerp(a, b, t):
    # Linear interpolation
    return a + t * (b - a)

def gradient(h, x, y):
    vectors = np.array([
        [1,1], [-1,1], [1,-1], [-1,-1],
        [1,0], [-1,0], [0,1], [0,-1]
    ])

    g = vectors[h % 8]

    return g[:,:,0] * x + g[:,:,1] * y

# ---------------------------------------------------
# Generate permutation table
# ---------------------------------------------------
p = np.arange(256, dtype=int)
np.random.shuffle(p)
p = np.stack([p, p]).flatten()

# ---------------------------------------------------
# Generate coordinate grid
# ---------------------------------------------------
lin = np.linspace(0, SCALE, SIZE, endpoint=False)

x, y = np.meshgrid(lin, lin)

# ---------------------------------------------------
# Multi-octave Perlin noise
# ---------------------------------------------------
noise = np.zeros((SIZE, SIZE), dtype=np.float32)

frequency = 1.0
amplitude = 1.0
max_amp = 0.0

for _ in range(OCTAVES):

    xf = x * frequency / SCALE 
    yf = y * frequency / SCALE 

    xi = xf.astype(int) & 255
    yi = yf.astype(int) & 255

    xf_frac = xf - np.floor(xf)
    yf_frac = yf - np.floor(yf)

    u = fade(xf_frac)
    v = fade(yf_frac)

    aa = p[p[xi] + yi]
    ab = p[p[xi] + yi + 1]
    ba = p[p[xi + 1] + yi]
    bb = p[p[xi + 1] + yi + 1]

    x1 = lerp(
        gradient(aa, xf_frac, yf_frac),
        gradient(ba, xf_frac - 1, yf_frac),
        u
    )

    x2 = lerp(
        gradient(ab, xf_frac, yf_frac - 1),
        gradient(bb, xf_frac - 1, yf_frac - 1),
        u
    )

    octave_noise = lerp(x1, x2, v)

    noise += octave_noise * amplitude

    max_amp += amplitude

    amplitude *= PERSISTENCE
    frequency *= LACUNARITY

# Normalize
noise /= max_amp

# ---------------------------------------------------
# Normalize to image range
# ---------------------------------------------------
noise = noise * 1.5
noise -= noise.min()
noise /= noise.max()

terrain = (noise * 255).astype(np.uint8)

# ---------------------------------------------------
# Optional smoothing
# ---------------------------------------------------
terrain = cv2.GaussianBlur(
    terrain,
    (5, 5),
    0
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------
cv2.imwrite(OUTPUT, terrain)

print(f"Saved huge terrain: {OUTPUT}")

print(terrain.min(), terrain.max())