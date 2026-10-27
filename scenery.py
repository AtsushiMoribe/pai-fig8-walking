"""
Scenery for the videos: a soil-like ground and a blue sky instead of Genesis' default checkerboard and dark background.

Only the look changes. The physics still uses the same plane.urdf as in training (its own visual is hidden),
so the robots walk exactly as before.
"""

import genesis as gs
import numpy as np

SKY_COLOR = (0.53, 0.76, 0.96)
TILE_M = 4.0  # one texture tile covers 4 m x 4 m


def soil_texture(size=512, seed=0):
    """Tileable soil image (uint8 RGB): band-passed noise for clods, fine grain, darker cracks and small pebbles."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(size)
    k = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2) + 1e-6

    def noise(lo, hi):
        # filtering white noise in Fourier space keeps the image periodic, so tiles join without seams
        spec = np.fft.fft2(rng.standard_normal((size, size))) * ((k > lo) & (k < hi)) / k**0.8
        n = np.real(np.fft.ifft2(spec))
        return (n - n.mean()) / n.std()

    clods = noise(1 / size, 12 / size)
    grain = noise(40 / size, 0.5)
    cracks = np.abs(noise(6 / size, 30 / size))
    shade = 0.55 * clods + 0.25 * grain - 0.35 * np.clip(0.4 - cracks, 0, None) * 4

    dark, light = np.array([92, 66, 44]), np.array([160, 122, 84])
    t = np.clip(0.5 + 0.18 * shade, 0, 1)[..., None]
    img = dark + (light - dark) * t

    # a few light pebbles
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(140):
        cy, cx, r = rng.integers(0, size), rng.integers(0, size), rng.uniform(1.5, 4.0)
        dy = (yy - cy + size // 2) % size - size // 2
        dx = (xx - cx + size // 2) % size - size // 2
        m = (dx**2 + dy**2) < r**2
        img[m] = img[m] * 0.4 + np.array([182, 168, 150]) * 0.6
    return np.clip(img, 0, 255).astype(np.uint8)


def vis_options(**kwargs):
    return gs.options.VisOptions(background_color=SKY_COLOR, ambient_light=(0.3, 0.3, 0.3), **kwargs)


def add_ground(scene):
    """Physics: the usual plane.urdf with its checkerboard hidden. Look: a soil-textured plane without collision."""
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True, visualization=False))
    scene.add_entity(
        gs.morphs.Plane(collision=False, plane_size=(200.0, 200.0), tile_size=(TILE_M, TILE_M)),
        surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(image_array=soil_texture())),
    )
