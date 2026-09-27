import math

import numpy as np

from .data_types import Object3D

try:
    from shapely import Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    from bbox import BBox3D
    from bbox.metrics import iou_3d
    HAS_BBOX = True
except ImportError:
    HAS_BBOX = False
    print("Warning: bbox library not installed. 3D IoU will use axis-aligned approximation.")


# Objects thinner than this (metres) along any axis get the 2-D projected IoU
# treatment instead of full 3-D IoU.
THIN_DIM_THRESHOLD = 0.1
_PARALLEL_TOLERANCE = math.sin(math.radians(5))
_DIST_TOLERANCE = 0.2


def _get_thin_face_corners(obj: Object3D) -> np.ndarray:
    """Return 4 face-centre points for the thinnest face of an OBB."""
    bbox = BBox3D(
        obj.position[0], obj.position[1], obj.position[2],
        obj.size[0], obj.size[1], obj.size[2],
        euler_angles=[0, 0, float(obj.angle_z)], is_center=True,
    )
    pts = bbox.p  # (8, 3)
    thin = int(np.argmin(obj.size))
    if thin == 0:
        return np.array([
            (pts[0] + pts[1]) / 2, (pts[2] + pts[3]) / 2,
            (pts[6] + pts[7]) / 2, (pts[4] + pts[5]) / 2,
        ])
    elif thin == 1:
        return np.array([
            (pts[0] + pts[3]) / 2, (pts[4] + pts[7]) / 2,
            (pts[5] + pts[6]) / 2, (pts[1] + pts[2]) / 2,
        ])
    else:
        return np.array([
            (pts[0] + pts[4]) / 2, (pts[1] + pts[5]) / 2,
            (pts[2] + pts[6]) / 2, (pts[3] + pts[7]) / 2,
        ])


def _are_planes_parallel_and_close(
    corners_1: np.ndarray,
    corners_2: np.ndarray,
    parallel_tolerance: float = _PARALLEL_TOLERANCE,
    dist_tolerance: float = _DIST_TOLERANCE,
) -> bool:
    p1, p2, p3, _ = corners_1
    q1, q2, q3, _ = corners_2
    n1 = np.cross(p2 - p1, p3 - p1)
    n2 = np.cross(q2 - q1, q3 - q1)
    n1_len = np.linalg.norm(n1)
    n2_len = np.linalg.norm(n2)
    if n1_len * n2_len < 1e-6:
        return False
    return (
        np.linalg.norm(np.cross(n1, n2)) / (n1_len * n2_len) < parallel_tolerance
        and abs(np.dot(q1 - p1, n1) / n1_len) < dist_tolerance
    )


def _calc_thin_bbox_iou_2d(
    corners_1: np.ndarray,
    corners_2: np.ndarray,
    parallel_tolerance: float = _PARALLEL_TOLERANCE,
    dist_tolerance: float = _DIST_TOLERANCE,
) -> float:
    """Project both thin-face quads onto a shared 2-D plane and compute IoU."""
    if not HAS_SHAPELY:
        return 0.0
    if not _are_planes_parallel_and_close(corners_1, corners_2, parallel_tolerance, dist_tolerance):
        return 0.0
    p1, p2, _, p4 = corners_2
    v1 = p2 - p1
    v2 = p4 - p1
    basis1 = v1 / np.linalg.norm(v1)
    basis1_orth = v2 - np.dot(v2, basis1) * basis1
    norm_orth = np.linalg.norm(basis1_orth)
    if norm_orth < 1e-9:
        return 0.0
    basis2 = basis1_orth / norm_orth

    def project(corners):
        return [[np.dot(pt - p1, basis1), np.dot(pt - p1, basis2)] for pt in corners]

    poly1 = Polygon(project(corners_1))
    poly2 = Polygon(project(corners_2))
    if not poly1.intersects(poly2):
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0


def compute_iou_3d_aa(obj1: Object3D, obj2: Object3D) -> float:
    """Compute axis-aligned 3D IoU between two objects."""
    min1 = obj1.position - obj1.size / 2
    max1 = obj1.position + obj1.size / 2
    min2 = obj2.position - obj2.size / 2
    max2 = obj2.position + obj2.size / 2

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_size)

    vol1 = np.prod(obj1.size)
    vol2 = np.prod(obj2.size)
    union_vol = vol1 + vol2 - inter_vol

    return inter_vol / union_vol if union_vol > 0 else 0.0


def compute_iou_3d(obj1: Object3D, obj2: Object3D) -> float:
    """Compute 3D IoU, with thin-object 2D fallback."""
    thin1 = np.min(obj1.size) < THIN_DIM_THRESHOLD
    thin2 = np.min(obj2.size) < THIN_DIM_THRESHOLD
    if (thin1 or thin2) and HAS_BBOX:
        c1 = _get_thin_face_corners(obj1)
        c2 = _get_thin_face_corners(obj2)
        return _calc_thin_bbox_iou_2d(c1, c2)
    if HAS_BBOX:
        bbox1 = BBox3D(
            obj1.position[0], obj1.position[1], obj1.position[2],
            obj1.size[0], obj1.size[1], obj1.size[2],
            euler_angles=[0, 0, float(obj1.angle_z)], is_center=True,
        )
        bbox2 = BBox3D(
            obj2.position[0], obj2.position[1], obj2.position[2],
            obj2.size[0], obj2.size[1], obj2.size[2],
            euler_angles=[0, 0, float(obj2.angle_z)], is_center=True,
        )
        return iou_3d(bbox1, bbox2)
    return compute_iou_3d_aa(obj1, obj2)
