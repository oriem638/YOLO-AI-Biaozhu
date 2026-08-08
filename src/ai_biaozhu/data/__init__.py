"""Public data-layer API.

Controller code should normally start with :func:`create_project` or
:func:`open_project` and keep the returned :class:`AnnotationProject` open.
"""

from .image_import import SUPPORTED_IMAGE_SUFFIXES, ImageImporter
from .project import (
    AnnotationProject,
    Project,
    create_project,
    open_project,
)
from .repository import AnnotationRepository, Repository
from .voc import (
    VocBox,
    VocCategoryPlan,
    VocDataset,
    VocImage,
    VocMergeDisposition,
    VocMergeItemPlan,
    VocMergeItemResult,
    VocMergePlan,
    VocMergeReport,
    VocProjectImport,
    create_project_from_voc,
    merge_voc_into_project,
    preflight_voc_merge,
    read_voc_dataset,
)
from .yolo import (
    YoloReadback,
    YoloReadbackImage,
    create_training_snapshot,
    export_yolo_detection,
    parse_yolo_detection,
    read_yolo_export,
    read_yolo_label,
    verify_yolo_export,
    yolo_boxes_to_pixels,
)

__all__ = [
    "AnnotationProject",
    "AnnotationRepository",
    "ImageImporter",
    "Project",
    "Repository",
    "SUPPORTED_IMAGE_SUFFIXES",
    "VocBox",
    "VocCategoryPlan",
    "VocDataset",
    "VocImage",
    "VocMergeDisposition",
    "VocMergeItemPlan",
    "VocMergeItemResult",
    "VocMergePlan",
    "VocMergeReport",
    "VocProjectImport",
    "YoloReadback",
    "YoloReadbackImage",
    "create_project",
    "create_project_from_voc",
    "create_training_snapshot",
    "export_yolo_detection",
    "merge_voc_into_project",
    "open_project",
    "parse_yolo_detection",
    "read_yolo_export",
    "read_yolo_label",
    "read_voc_dataset",
    "preflight_voc_merge",
    "verify_yolo_export",
    "yolo_boxes_to_pixels",
]
