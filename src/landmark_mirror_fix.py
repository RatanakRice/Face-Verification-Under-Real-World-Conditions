"""
Training-free occlusion correction: fits a RANSAC affine mirror transform
between visible left/right MediaPipe landmark pairs and pastes the visible
half of the face over the occluded region, falling back to OpenCV inpainting
if the fit is unreliable.

Input: a folder of occluded images, the mesh pairing table (--build-pairs
builds this once from a sample of clean images), and optionally a
boxes.json of true occluder coordinates (if omitted, find_box() detects the
occluder itself from its dark pixels -- this is the path quality_gate.py
uses, since it has no ground-truth box).
Output: a folder of repaired images.
Report section: Methodology / Mitigation Methods (mirror-fill).
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

# mediapipe >=1.0 dropped the legacy `solutions` API this was originally
# written against; this uses the newer Tasks API instead, same underlying
# face-mesh model and 468-point topology. Needs Google's public model bundle
# at MODEL_PATH (see the "download the model" note in the project README).
MODEL_PATH = "model/face_landmarker.task"

_FM = mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
    base_options=mp_base_options.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
))

# silhouette points slide along the face edge when the head turns, so they
# are not true mirror partners; they are left out of the fit
OVAL = {i for c in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL for i in (c.start, c.end)}

# Every tunable knob in one place, grouped by which step above it belongs to.
DEFAULTS = dict(
    # step 3 -- fit the mirror
    num_pairs=30,        # how many of the closest-to-the-box landmark pairs to fit on
    edge_margin=4,        # px a landmark must sit clear of every occluder to count as "visible"
    ransac_frac=0.015,    # RANSAC inlier threshold, as a fraction of image width

    # step 4 -- nudge the fit to pixel accuracy
    nudge_frac=0.04,      # search radius for the nudge, as a fraction of image size
    nudge_prior=0.01,     # mild penalty per pixel of shift, breaks ties toward "no shift"
    nudge_step=2,         # coarse grid step (then refined to 1 px)

    # step 5 -- match brightness across the seam
    rim_width=4,          # px just outside the box used to measure the colour mismatch

    # step 6 -- paste, inpaint, blend
    box_grow=2,           # px the pasted region extends past the original box
    inpaint_blend=6,      # px over which a botched patch fades to plain inpainting
    edge_feather=4,       # px over which the box edge blends into the untouched photo
)


# ---------------- box and masks ----------------
def find_box(img, thresh=20):
    """
    Largest solid black rectangle in the image that doesn't touch the border
    -- the box occlusion.py drew. Classic "largest rectangle in a histogram"
    trick, applied one image row at a time, so it only matches a genuinely
    solid block and not just any dark region (dark hair, sunglasses, ...).
    """
    is_dark = cv2.medianBlur((img.max(axis=2) < thresh).astype(np.uint8), 3)
    H, W = is_dark.shape
    heights = np.zeros(W, int)
    best = (0, 0, 0, 0, 0)  # (area, x, y, w, h)
    for y in range(H):
        heights = np.where(is_dark[y] > 0, heights + 1, 0)
        stack = []
        for x in range(W + 1):
            h = heights[x] if x < W else 0
            start = x
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                touches_border = s == 0 or x == W or y - sh + 1 == 0 or y == H - 1
                if not touches_border and sh * (x - s) > best[0]:
                    best = (sh * (x - s), s, y - sh + 1, x - s, sh)
                start = s
            stack.append((start, h))
    _, x, y, w, h = best
    return int(x), int(y), int(w), int(h)


def box_mask(shape, box, grow=0):
    """Binary mask of the box, optionally grown by `grow` px on every side."""
    x, y, w, h = box
    mask = np.zeros(shape[:2], np.uint8)
    mask[max(0, y - grow):y + h + grow, max(0, x - grow):x + w + grow] = 1
    return mask


def ring(shape, box, inner, outer, occ):
    """Pixel band between `inner` and `outer` px outside the box, clear of
    every occluder (the box itself and any other hidden area)."""
    band = (box_mask(shape, box, outer) - box_mask(shape, box, inner)).astype(bool)
    clear_of_occluders = (occ == 0) & (cv2.dilate(occ, np.ones((9, 9), np.uint8)) == 0)
    return band & clear_of_occluders


# ---------------- step 1: face points ----------------
def face_points(img, min_side=192):
    """Run MediaPipe FaceLandmarker on img, returning all 468 mesh points in
    original-image pixel coordinates, or None if no face is found."""
    H, W = img.shape[:2]
    scale = max(1.0, min_side / min(H, W))          # enlarge small crops
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else img
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
    result = _FM.detect(mp_image)
    if not result.face_landmarks:
        return None
    bh, bw = big.shape[:2]
    # the Tasks API always appends 10 iris points (478 total) with no way to
    # disable them; drop them to match the 468-point topology used below
    pts = np.array([[p.x * bw, p.y * bh] for p in result.face_landmarks[0][:468]])
    return pts / scale                              # scale back


def detect_landmarks(img, occ):
    """Normal detection; if that finds no face, retry on a copy with the black
    areas pre-inpainted (solid blocks can hide the face from the detector)."""
    pts = face_points(img)
    if pts is None:
        preview = cv2.inpaint(img, cv2.dilate(occ, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
        pts = face_points(preview)
    return pts


# ---------------- step 2: left/right pairs (built once) ----------------
def build_pairs(images):
    """
    Landmark i of a flipped face sits where partner(i) sits on the real face.
    Detect both on each sample image, match the two point sets (summed over
    all samples) to find that pairing. Run once via build_pairs_and_save();
    fix() reuses the saved table for every image afterwards.
    """
    cost = np.zeros((468, 468))
    for img in images:
        W = img.shape[1]
        pts = face_points(img)
        flipped_pts = face_points(cv2.flip(img, 1))
        if pts is None or flipped_pts is None:
            continue
        flipped_pts[:, 0] = W - 1 - flipped_pts[:, 0]
        cost += np.linalg.norm(pts[:, None] - flipped_pts[None], axis=2) / np.ptp(pts[:, 0])
    _, pairs = linear_sum_assignment(cost.T)     # pairs[i] = partner of i
    return pairs


def build_pairs_and_save(input_dir: Path, output_path: Path, num_samples: int = 40, seed: int = 0):
    """Sample a handful of clean faces, learn the left/right landmark pairing
    (see build_pairs), and save it to disk. Run this once per project -- the
    result doesn't depend on which photo fix() is later applied to."""
    image_paths = sorted(input_dir.glob("*.jpg"))
    sample_paths = random.Random(seed).sample(image_paths, min(num_samples, len(image_paths)))
    images = [img for img in (cv2.imread(str(p)) for p in sample_paths) if img is not None]
    pairs = build_pairs(images)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, pairs)
    print(f"Saved pairing table ({len(pairs)} points, from {len(images)} sample faces) to {output_path}")


# ---------------- step 3: fit the mirror ----------------
def dist_to_box(pts, box):
    """Distance from each point to the box's edge (0 for points inside it)."""
    x, y, w, h = box
    dx = np.maximum(np.maximum(x - pts[:, 0], pts[:, 0] - (x + w)), 0)
    dy = np.maximum(np.maximum(y - pts[:, 1], pts[:, 1] - (y + h)), 0)
    return np.hypot(dx, dy)


def select_pairs(pts, pairs, box, occ, W, c):
    """
    Pick the best landmark pairs to fit the mirror on: both points visible
    (clear of the box and of any other hidden area), not on the centre line
    (those barely move when mirrored), and on opposite sides of the box (i
    near, j far) so the fit is well-constrained. Pairs closest to the box are
    used first since they matter most for a seamless patch.
    """
    H = occ.shape[0]
    dist = dist_to_box(pts, box)
    visible = dist > c['edge_margin']
    other_hidden = occ.copy()
    other_hidden[box_mask(occ.shape, box) > 0] = 0
    if other_hidden.any():                              # other hidden areas, if any
        dt = cv2.distanceTransform((other_hidden == 0).astype(np.uint8), cv2.DIST_L2, 5)
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        visible &= dt[yi, xi] > c['edge_margin']
    candidates = []
    for i, j in enumerate(pairs):
        if i == j or not (visible[i] and visible[j]) or i in OVAL or j in OVAL:
            continue
        if abs(pts[i, 0] - pts[j, 0]) < 0.08 * W:        # centre-line points
            continue
        if dist[j] <= dist[i]:                           # j must be the far side
            continue
        candidates.append((dist[i], i, j))
    candidates.sort()
    use = candidates[:c['num_pairs']]
    dst = np.array([pts[i] for _, i, _ in use], np.float32).reshape(-1, 2)
    src = np.array([pts[j] for _, _, j in use], np.float32).reshape(-1, 2)
    src[:, 0] = W - 1 - src[:, 0]                        # flip, then move/rotate/stretch
    return src, dst


def mirror(img, M):
    """
    Warp the horizontally-flipped image by M onto the occluded side. M maps
    flipped-image points to their partner points on the box side (fit in
    select_pairs), so warping the flipped image by M directly produces the
    mirrored patch -- OpenCV inverts M internally to do the pixel lookup.
    """
    H, W = img.shape[:2]
    return cv2.warpAffine(cv2.flip(img, 1), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def source_valid(occ, face, M, grow):
    """
    True where the mirrored pixel came from visible skin: not from a hidden
    area and not from outside the face. Uses the same M as mirror() so the
    two stay in sync.
    """
    H, W = occ.shape[:2]
    occ_grown = cv2.dilate(occ, np.ones((2 * grow + 1,) * 2, np.uint8))
    visible_skin = cv2.flip((1 - occ_grown).astype(np.uint8) & face, 1)
    return cv2.warpAffine(visible_skin, M, (W, H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0


# ---------------- step 4: nudge ----------------
def sobel_grads(img):
    """Horizontal and vertical Sobel gradients of the grayscale image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)


def nudge(img, mirrored, valid, box, occ, c):
    """
    Small shift whose edges best correlate with the real edges around the
    box. Correlation ignores brightness and contrast; a mild penalty prefers
    small shifts. Coarse grid first, then 1 px refinement.
    """
    H, W = img.shape[:2]
    radius = max(1, int(round(c['nudge_frac'] * max(H, W))))
    band = ring(img.shape, box, 4, 12, occ)
    photo_gx, photo_gy = sobel_grads(img)
    mirror_gx, mirror_gy = sobel_grads(mirrored)

    def cost(dx, dy):
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_valid = cv2.warpAffine(valid.astype(np.uint8), shift, (W, H), flags=cv2.INTER_NEAREST) > 0
        sel = band & shifted_valid
        if sel.sum() < 0.5 * band.sum():
            return np.inf
        gx = cv2.warpAffine(mirror_gx, shift, (W, H), borderMode=cv2.BORDER_REFLECT)
        gy = cv2.warpAffine(mirror_gy, shift, (W, H), borderMode=cv2.BORDER_REFLECT)
        a = np.r_[gx[sel], gy[sel]]
        b = np.r_[photo_gx[sel], photo_gy[sel]]
        correlation = -np.dot(a - a.mean(), b - b.mean()) / (
            np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()) + 1e-6)
        return correlation + c['nudge_prior'] * np.hypot(dx, dy)

    step = c['nudge_step']
    grid = [(dx, dy) for dy in range(-radius, radius + 1, step) for dx in range(-radius, radius + 1, step)]
    best = min(((cost(dx, dy), dx, dy) for dx, dy in grid), key=lambda t: t[0])
    if step > 1 and np.isfinite(best[0]):
        for _ in range(step):
            around = [(best[1] + i, best[2] + j) for i in (-1, 0, 1) for j in (-1, 0, 1)
                      if abs(best[1] + i) <= radius and abs(best[2] + j) <= radius]
            neighbour_best = min(((cost(dx, dy), dx, dy) for dx, dy in around), key=lambda t: t[0])
            if neighbour_best[0] >= best[0]:
                break
            best = neighbour_best
    if not np.isfinite(best[0]):
        return 0, 0
    return best[1], best[2]


# ---------------- step 5: match brightness ----------------
def harmonic(values, known, unknown):
    """Solve Laplace's equation on `unknown` pixels with `known` pixels fixed:
    the smoothest field that agrees with the known border values."""
    H, W = known.shape
    idx = -np.ones((H, W), int)
    ys, xs = np.nonzero(unknown)
    n = len(ys)
    idx[ys, xs] = np.arange(n)
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    b = np.zeros((n, values.shape[2]))
    for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy, xx = ys + oy, xs + ox
        inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        yy_c, xx_c = np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)
        nb_unk = inb & unknown[yy_c, xx_c]
        nb_kn = inb & ~unknown[yy_c, xx_c] & known[yy_c, xx_c]
        k = np.nonzero(nb_unk)[0]
        rows += list(k); cols += list(idx[yy_c[k], xx_c[k]]); vals += [-1.0] * len(k)
        k = np.nonzero(nb_kn)[0]
        b[k] += values[yy_c[k], xx_c[k]]
        diag += nb_unk | nb_kn
    rows += list(range(n)); cols += list(range(n)); vals += list(np.maximum(diag, 1))
    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    out = np.zeros_like(values)
    out[ys, xs] = spsolve(A, b).reshape(n, -1)
    return out


def offset_brightness(img, mirrored, valid, box, occ, face):
    """v1 method: one colour shift from skin-like ring pixels (fallback only)."""
    band = ring(img.shape, box, 4, 10, occ) & valid & (face > 0)
    if band.sum() < 5:
        return mirrored
    lab_all = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    fm = (face > 0) & (cv2.dilate(occ, np.ones((13, 13), np.uint8)) == 0)
    skin_ab = np.median(lab_all[fm][:, 1:].astype(np.float32), axis=0)
    lr = lab_all[band].astype(np.float32)
    lm = cv2.cvtColor(mirrored, cv2.COLOR_BGR2LAB)[band].astype(np.float32)
    keep = (np.linalg.norm(lr[:, 1:] - skin_ab, axis=1) < 10) & \
           (np.linalg.norm(lm[:, 1:] - skin_ab, axis=1) < 10)
    d = img[band].astype(np.float32) - mirrored[band].astype(np.float32)
    if keep.sum() > 30:
        d = d[keep]
    med = np.median(d, axis=0)
    good = np.all(np.abs(d - med) < 25, axis=1)
    if good.sum() > 20:
        d = d[good]
    return np.clip(mirrored.astype(np.float32) + np.clip(np.median(d, axis=0), -60, 60),
                   0, 255).astype(np.uint8)


def match_brightness(img, mirrored, valid, box, occ, face, c):
    """
    Colour difference measured on a thin ring just past the JPEG halo, then
    spread smoothly across the box. Handles lighting that changes across the
    patch, which a single offset cannot.
    """
    rim_start = c['rim_width']
    rim = ring(img.shape, box, rim_start, rim_start + 2, occ) & valid
    if rim.sum() < 20:                       # nothing reliable around the box
        return offset_brightness(img, mirrored, valid, box, occ, face)
    diff = img.astype(np.float32) - mirrored.astype(np.float32)
    unknown = box_mask(img.shape, box, rim_start).astype(bool) & ~rim
    field = harmonic(diff, rim, unknown)
    field[rim] = diff[rim]
    return np.clip(mirrored.astype(np.float32) + field, 0, 255).astype(np.uint8)


# ---------------- full pipeline ----------------
def fix(img, pairs, box=None, occ=None, cfg=None, pts=None, debug=False):
    """
    Repair one occluded face. img: BGR uint8. box: (x, y, w, h), auto-detected
    from the solid black region if None. occ: mask of every hidden pixel,
    defaults to just the box. Returns the repaired image (and a dict of
    diagnostics if debug=True).
    """
    c = dict(DEFAULTS, **(cfg or {}))
    H, W = img.shape[:2]
    box = box or find_box(img)
    if occ is None:
        occ = box_mask(img.shape, box)
    region = box_mask(img.shape, box, c['box_grow'])
    info = dict(box=box, method='inpaint fallback')

    def done(out):
        return (out, info) if debug else out

    plain_inpaint = lambda: cv2.inpaint(img, region, 5, cv2.INPAINT_TELEA)

    # step 1: find the face, even if the box is hiding it from the detector
    if pts is None:
        pts = detect_landmarks(img, occ)
    if pts is None:
        info['reason'] = 'no face found'
        return done(plain_inpaint())
    face = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(face, cv2.convexHull(pts.astype(np.int32)), 1)

    # step 3: fit the mirror using landmark pairs clear of every occluder
    src, dst = select_pairs(pts, pairs, box, occ, W, c)
    if len(src) < 6:
        info['reason'] = 'too few visible pairs'
        return done(plain_inpaint())
    M, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                      ransacReprojThreshold=c['ransac_frac'] * W, confidence=0.999)
    if M is None:
        info['reason'] = 'fit failed'
        return done(plain_inpaint())

    # step 4: nudge the fit to pixel-accuracy; step 5: match the brightness
    mirrored = mirror(img, M)
    valid = source_valid(occ, face, M, c['box_grow'])
    dx, dy = nudge(img, mirrored, valid, box, occ, c)
    if dx or dy:
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        mirrored = cv2.warpAffine(mirrored, shift, (W, H), borderMode=cv2.BORDER_REFLECT)
        valid = cv2.warpAffine(valid.astype(np.uint8), shift, (W, H), flags=cv2.INTER_NEAREST) > 0
    mirrored = match_brightness(img, mirrored, valid, box, occ, face, c)

    # step 6: paste the mirror in, inpaint anything it couldn't cover, feather the edge
    reg = region.astype(bool)
    out = img.copy()
    out[reg] = mirrored[reg]
    unusable = (reg & ~valid).astype(np.uint8)
    if unusable.any():
        out = cv2.inpaint(out, unusable, 3, cv2.INPAINT_TELEA)
    if c['inpaint_blend'] > 0:
        backup = cv2.inpaint(img, region, 5, cv2.INPAINT_TELEA)
        dist_in = cv2.distanceTransform(region, cv2.DIST_L2, 3)
        weight = np.clip(dist_in / c['inpaint_blend'], 0, 1)[..., None]
        out = np.where(reg[..., None], weight * out + (1 - weight) * backup, out).astype(np.uint8)
    if c['edge_feather'] > 0:
        dist_out = cv2.distanceTransform((~reg).astype(np.uint8), cv2.DIST_L2, 3)
        weight = np.clip(1 - dist_out / c['edge_feather'], 0, 1)[..., None]
        weight[reg] = 1
        out = (weight * out + (1 - weight) * img).astype(np.uint8)

    info.update(method='mirror', pairs=len(src), inliers=int(inliers.sum()), nudge=(dx, dy),
                mirrored_frac=float(valid[reg].mean()), M=M)
    return done(out)


# ---------------- batch processing ----------------
def process_folder(image_paths: list[Path], output_dir: Path, pairs, boxes: dict = None):
    """
    Mirror-fix every image, writing to output_dir. boxes maps filename ->
    {"box": [x, y, w, h], ...} from occlusion.py's boxes.json, giving the
    exact occlusion box; if omitted, each box is auto-detected instead
    (needed for real, non-synthetic occluders like sunglasses).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    method_counts = {}
    skipped = []
    for image_path in tqdm(image_paths, desc="Mirror-fixing", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append((image_path.name, "unreadable"))
            continue

        box = tuple(boxes[image_path.name]["box"]) if boxes else None
        fixed, info = fix(image, pairs, box=box, debug=True)
        cv2.imwrite(str(output_dir / image_path.name), fixed)
        method_counts[info['method']] = method_counts.get(info['method'], 0) + 1

    print(f"Done. {len(skipped)} images skipped. Output: {output_dir}")
    for name, reason in skipped[:20]:
        print(f"  skipped: {name} ({reason})")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more")
    print(f"Method counts: {method_counts}")


def main():
    parser = argparse.ArgumentParser(description="Repair a synthetic occlusion box with the mirrored other half of the same face.")
    parser.add_argument("--input", required=True, help="Fix mode: occluded crops folder. Build-pairs mode: clean crops folder to learn the pairing from.")
    parser.add_argument("--output", help="Fix mode: output folder for the repaired images.")
    parser.add_argument("--boxes", help="Fix mode: boxes.json from occlusion.py. Omit to auto-detect each box.")
    parser.add_argument("--pairs", default="model/mesh_pairs.npy", help="Landmark pairing table, built by --build-pairs.")
    parser.add_argument("--build-pairs", action="store_true", help="Build the pairing table from --input and save it to --pairs, instead of running the fix.")
    parser.add_argument("--num-samples", type=int, default=40, help="Build-pairs mode: how many images to sample. Default: 40.")
    args = parser.parse_args()

    if args.build_pairs:
        build_pairs_and_save(Path(args.input), Path(args.pairs), args.num_samples)
        return

    if not args.output:
        parser.error("--output is required unless --build-pairs is set")

    pairs = np.load(args.pairs)
    boxes = json.load(open(args.boxes)) if args.boxes else None
    image_paths = sorted(Path(args.input).glob("*.jpg"))
    print(f"Found {len(image_paths)} images in {args.input}")
    process_folder(image_paths, Path(args.output), pairs, boxes)


if __name__ == "__main__":
    main()

# Usage:
#   Build the pairing table once (needed before the first real fix run):
#     python src/landmark_mirror_fix.py --input data2/processed/aligned/lfw_bigger --build-pairs --pairs model/mesh_pairs.npy
#   Then repair every occluded image in a folder:
#     python src/landmark_mirror_fix.py --input data2/processed/occlusion/lfw_bigger/block_40 --boxes data2/processed/occlusion/lfw_bigger/block_40/boxes.json --output data2/processed/occlusion_fixed/lfw_bigger/block_40
