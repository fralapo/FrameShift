"""Utilities for face and object detection."""
from typing import List, Tuple, Dict, Any, Set, Optional
import cv2
from pathlib import Path
import requests
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import hashlib
import os
import logging

logger = logging.getLogger('frameshift.utils.detection')

# Network timeout for model downloads (seconds). Without this requests can hang
# forever on a stalled connection.
MODEL_DOWNLOAD_TIMEOUT = 60

# Pinned model sources with SHA-256 digests of the trusted release artifacts.
#
# A PyTorch ".pt" file is a pickle archive; YOLO() ultimately unpickles it, which
# is equivalent to executing arbitrary code. Every file is therefore verified
# against the digest below BEFORE it is ever handed to YOLO() -- whether it was
# just downloaded or was already sitting in the models/ directory. A mismatched
# or corrupt file is deleted and never loaded.
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "yolo11n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    },
    "yolov11n-face.pt": {
        "url": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt",
        "sha256": "9420a9d4933940d2202f511d45df1750e3c1546dd9efc7a3985caae3bbbf7ed2",
    },
}


def _sha256_of_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks to bound memory use."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_verified_model(filename: str, model_dir: Path) -> Optional[Path]:
    """
    Return a path to a SHA-256-verified local copy of ``filename``, or ``None``
    if it cannot be obtained and verified.

    The returned path is safe to hand to ``YOLO()``: an existing local file is
    trusted only when its digest matches the pinned value, and a freshly
    downloaded file is verified before being moved into place. Any file that
    fails verification is removed rather than loaded.
    """
    if filename not in MODEL_REGISTRY:
        logger.error(f"No pinned source/digest for model '{filename}'. Refusing to load.")
        return None

    spec = MODEL_REGISTRY[filename]
    expected = spec["sha256"].lower()
    dest = model_dir / filename

    # Trust an already-present file only if its digest matches.
    if dest.is_file():
        try:
            actual = _sha256_of_file(dest)
        except OSError as e:
            logger.error(f"Could not read local '{dest}' for verification: {e}")
            return None
        if actual.lower() == expected:
            logger.info(f"Verified existing model '{filename}'.")
            return dest
        logger.warning(
            f"Local '{filename}' failed integrity check (expected {expected[:12]}..., "
            f"got {actual[:12]}...). Discarding and re-downloading."
        )
        try:
            dest.unlink()
        except OSError as e:
            logger.error(f"Could not remove untrusted file '{dest}': {e}. Refusing to load.")
            return None

    # Download to a temporary file, verify, then atomically move into place so a
    # partial or tampered download can never be picked up on a later run.
    tmp = dest.with_name(dest.name + ".part")
    logger.info(f"Downloading '{filename}' from {spec['url']} ...")
    try:
        with requests.get(spec["url"], stream=True, timeout=MODEL_DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        logger.error(f"Download of '{filename}' failed: {e}", exc_info=True)
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return None

    try:
        actual = _sha256_of_file(tmp)
    except OSError as e:
        logger.error(f"Could not read downloaded '{tmp}' for verification: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass
        return None

    if actual.lower() != expected:
        logger.error(
            f"Downloaded '{filename}' failed integrity check (expected {expected[:12]}..., "
            f"got {actual[:12]}...). Discarding."
        )
        try:
            tmp.unlink()
        except OSError:
            pass
        return None

    try:
        os.replace(tmp, dest)
    except OSError as e:
        logger.error(f"Could not finalize '{dest}': {e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return None

    logger.info(f"Verified and installed '{filename}'.")
    return dest


class Detector:
    """
    Handles face and object detection.
    Uses a specialized YOLOv8 model for face detection (from Hugging Face)
    and a general YOLOv8n model for other objects (conditionally).
    MediaPipe is used as a fallback for face detection if the YOLO face model fails to load.
    """

    def __init__(self, yolo_face_conf: float = 0.3, yolo_obj_conf: float = 0.25, mp_face_conf: float = 0.5):
        self.yolo_face_conf = yolo_face_conf
        self.yolo_obj_conf = yolo_obj_conf
        self.mp_face_conf = mp_face_conf

        # Define model directory
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load general object detection model (yolo11n.pt). ensure_verified_model
        # downloads if needed and only returns a path whose SHA-256 matches the
        # pinned digest, so an untrusted/corrupt file is never loaded.
        self.obj_model = None
        obj_model_path = ensure_verified_model("yolo11n.pt", model_dir)
        if obj_model_path is not None:
            try:
                self.obj_model = YOLO(str(obj_model_path))
                logger.info(f"Successfully loaded general object model from {obj_model_path}.")
            except Exception as e_load_obj:
                logger.error(f"Could not load general object model from {obj_model_path}: {e_load_obj}. Object detection might not work.", exc_info=True)
                self.obj_model = None
        else:
            logger.error("General object model unavailable (download or integrity check failed). Object detection will be unavailable.")


        # Attempt to load specialized YOLO face detection model (yolov11n-face.pt)
        self.yolo_face_model = None
        self.mp_face_model = None

        face_model_path = ensure_verified_model("yolov11n-face.pt", model_dir)
        if face_model_path is not None:
            try:
                logger.info(f"Attempting to load YOLO face model from {face_model_path}...")
                self.yolo_face_model = YOLO(str(face_model_path))
                logger.info(f"Successfully loaded YOLO face model from {face_model_path}.")
            except Exception as e_load_face:
                logger.warning(f"Could not load YOLO face model from {face_model_path}: {e_load_face}")
                self.yolo_face_model = None
        else:
            logger.warning("YOLO face model unavailable (download or integrity check failed). Face detection with YOLO will be unavailable.")
            self.yolo_face_model = None


        # If self.yolo_face_model is still None after attempts, fallback to MediaPipe
        if self.yolo_face_model is None:
            logger.warning(f"YOLO face model not loaded. Falling back to MediaPipe for face detection.")
            try:
                self.mp_face_model = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=self.mp_face_conf
                )
                logger.info("MediaPipe Face Detection initialized as fallback.")
            except Exception as e_mp:
                logger.error(f"Could not initialize MediaPipe Face Detection: {e_mp}", exc_info=True)
                self.mp_face_model = None



    def detect(self, frame: np.ndarray, active_object_labels: Set[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Return lists of face detections and object detections.
        Faces are always detected (YOLOv8-Face or MediaPipe fallback).
        Other objects are detected by YOLOv8n only if active_object_labels is not empty.
        Each detection is a dict: {'box': (x1,y1,x2,y2), 'label': str, 'confidence': float}
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # For MediaPipe if used

        faces_detected: List[Dict[str, Any]] = []

        # 1. Face Detection
        if self.yolo_face_model:
            try:
                yolo_face_preds = self.yolo_face_model.predict(frame, imgsz=320, conf=self.yolo_face_conf, verbose=False)
                for res in yolo_face_preds:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    confidences = res.boxes.conf.cpu().numpy()
                    # class_ids = res.boxes.cls.cpu().numpy().astype(int)
                    # Assuming class 0 is 'face' for this model, or it only has one class.
                    # For arnabdhar/YOLOv8-Face-Detection, it's typically class 0: 'face'.
                    # yolo_face_class_names = res.names if hasattr(res, 'names') and res.names else {0: 'face'}

                    for i in range(len(boxes)):
                        box_coords = tuple(boxes[i].astype(int))
                        # label = yolo_face_class_names.get(class_ids[i], 'face') # Ensure label is 'face'
                        faces_detected.append({
                            'box': box_coords,
                            'label': 'face', # Standardize label
                            'confidence': confidences[i]
                        })
            except Exception as e_yolo_face:
                logger.error(f"YOLOv8-Face-Detection predict failed: {e_yolo_face}. Attempting MediaPipe fallback if available.", exc_info=True)
                if self.mp_face_model:
                    self.yolo_face_model = None
                else:
                     faces_detected = []

        if not self.yolo_face_model and self.mp_face_model:
            try:
                mp_results = self.mp_face_model.process(rgb_frame)
                if mp_results.detections:
                    for det in mp_results.detections:
                        rel_box = det.location_data.relative_bounding_box
                        x1 = int(rel_box.xmin * w)
                        y1 = int(rel_box.ymin * h)
                        bw = int(rel_box.width * w)
                        bh = int(rel_box.height * h)
                        faces_detected.append({
                            'box': (x1, y1, x1 + bw, y1 + bh),
                            'label': 'face',
                            'confidence': det.score[0] if det.score else self.mp_face_conf
                        })
            except Exception as e_mp_face:
                 logger.error(f"MediaPipe face detection failed: {e_mp_face}", exc_info=True)
                 faces_detected = []


        # 2. Object Detection (Conditional)
        objects_detected: List[Dict[str, Any]] = []
        if self.obj_model and active_object_labels:
            try:
                yolo_obj_preds = self.obj_model.predict(frame, imgsz=320, conf=self.yolo_obj_conf, verbose=False)
                for res in yolo_obj_preds:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    confidences = res.boxes.conf.cpu().numpy()
                    class_ids = res.boxes.cls.cpu().numpy().astype(int)
                    class_names_map = res.names if hasattr(res, 'names') and res.names else self.obj_model.names

                    for i in range(len(boxes)):
                        label = class_names_map.get(class_ids[i], f"class_{class_ids[i]}")
                        if label in active_object_labels: # Filter by active labels
                            box_coords = tuple(boxes[i].astype(int))
                            objects_detected.append({
                                'box': box_coords,
                                'label': label,
                                'confidence': confidences[i]
                            })
            except Exception as e_yolo_obj:
                logger.error(f"YOLOv8n object detection predict failed: {e_yolo_obj}", exc_info=True)
                objects_detected = []

        return faces_detected, objects_detected
