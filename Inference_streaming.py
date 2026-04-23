import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import time
import torch
import torchaudio
import numpy as np
import cv2
from collections import defaultdict
import yaml

from look2hear.models import Dolphin
from look2hear.datas.transform import get_preprocessing_pipelines

# Auto-select face landmark backend: mediapipe (fast, no TF) > opencv (always available) > face_alignment (slow, TF)
_USE_MEDIAPIPE = False
_USE_OPENCV = False
_mp_face_detection_cls = None
_mp_face_mesh_cls = None

# Try mediapipe with multiple import paths (different versions have different APIs)
for _attempt in range(4):
    try:
        if _attempt == 0:
            from mediapipe import solutions as _mp_solutions
            _mp_face_detection_cls = _mp_solutions.face_detection.FaceDetection
            _mp_face_mesh_cls = _mp_solutions.face_mesh.FaceMesh
        elif _attempt == 1:
            from mediapipe.python.solutions import face_detection as _mp_fd_mod
            from mediapipe.python.solutions import face_mesh as _mp_fm_mod
            _mp_face_detection_cls = _mp_fd_mod.FaceDetection
            _mp_face_mesh_cls = _mp_fm_mod.FaceMesh
        elif _attempt == 2:
            import mediapipe as _mp
            _mp_face_detection_cls = _mp.solutions.face_detection.FaceDetection
            _mp_face_mesh_cls = _mp.solutions.face_mesh.FaceMesh
        elif _attempt == 3:
            # Some builds expose solutions under different paths
            from mediapipe.solutions import face_detection as _mp_fd_mod2
            from mediapipe.solutions import face_mesh as _mp_fm_mod2
            _mp_face_detection_cls = _mp_fd_mod2.FaceDetection
            _mp_face_mesh_cls = _mp_fm_mod2.FaceMesh
        _USE_MEDIAPIPE = True
        break
    except (ImportError, AttributeError, ModuleNotFoundError):
        continue

if not _USE_MEDIAPIPE:
    # Fallback: OpenCV Haar Cascade (always available, no TF/mediapipe dependency)
    _USE_OPENCV = True
    print("[INFO] mediapipe not available, using OpenCV Haar Cascade for face detection")


# ============================================
# Timer & Stats utilities (unchanged)
# ============================================
class Timer:
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time

    def __str__(self):
        return f"{self.name}: {self.elapsed*1000:.2f} ms"


class SeparatorTimingStats:
    def __init__(self):
        self.chunk_times = []
        self.chunk_durations = []
        self.rtf_list = []
        self.last_chunk_latency = 0.0

    def record_chunk(self, infer_time_sec, audio_duration_sec):
        self.chunk_times.append(infer_time_sec)
        self.chunk_durations.append(audio_duration_sec)
        rtf = infer_time_sec / audio_duration_sec if audio_duration_sec > 0 else 0
        self.rtf_list.append(rtf)
        self.last_chunk_latency = infer_time_sec

    def summary(self):
        if not self.chunk_times:
            return
        print("\n" + "="*80)
        print("Separator Inference Timing Stats")
        print("="*80)
        print(f"\n{'Chunk':<8} {'Infer(ms)':<15} {'Audio(ms)':<15} {'RTF':<10}")
        print("-"*80)
        for i, (t, d, r) in enumerate(zip(self.chunk_times, self.chunk_durations, self.rtf_list)):
            print(f"{i+1:<8} {t*1000:>13.2f}   {d*1000:>13.2f}   {r:>8.3f}")
        print("-"*80)
        total_infer = sum(self.chunk_times)
        total_audio = sum(self.chunk_durations)
        print(f"\n  Total inference: {total_infer*1000:.2f}ms ({total_infer:.3f}s)")
        print(f"  Total audio:     {total_audio*1000:.2f}ms ({total_audio:.3f}s)")
        print(f"  Avg RTF:         {np.mean(self.rtf_list):.3f}x")
        print(f"  Last chunk lat:  {self.last_chunk_latency*1000:.2f}ms")
        print(f"  Chunks:          {len(self.chunk_times)}")
        print("="*80)


class ModuleTimingStats:
    def __init__(self):
        self.stats = defaultdict(list)

    def record(self, name, elapsed_sec):
        self.stats[name].append(elapsed_sec)

    def summary(self):
        print("\n" + "="*60)
        print("Module Timing Stats")
        print("="*60)
        for name, times in self.stats.items():
            total = sum(times)
            print(f"{name:30s}: {total*1000:8.1f} ms ({total:.3f}s)")
        print("="*60)


# ============================================
# Helper functions
# ============================================


def convert_bgr2gray(data):
    return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in data], axis=0)


def face2head(boxes, scale=1.5):
    new_boxes = []
    for box in boxes:
        width = box[2] - box[0]
        height = box[3] - box[1]
        width_center = (box[2] + box[0]) / 2
        height_center = (box[3] + box[1]) / 2
        square_width = int(max(width, height) * scale)
        new_box = [width_center - square_width/2, height_center - square_width/2,
                   width_center + square_width/2, height_center + square_width/2]
        new_boxes.append(new_box)
    return new_boxes


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# ============================================
# Streaming Face Processor
# ============================================

# Mediapipe lip landmark indices for mouth region extraction
_MP_LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]


class StreamingFaceProcessor:
    """Streaming face detection, landmark tracking, and mouth ROI extraction.

    Supports two backends:
    - mediapipe (default, fast, no TensorFlow dependency, landmark-based mouth crop)
    - opencv (fallback, Haar Cascade + heuristic mouth region, no extra deps)

    Processes video frames incrementally (per chunk) instead of loading the
    entire video. Runs heavy face detection every N frames; for intermediate
    frames, reuses previous detection results.
    """

    def __init__(self, num_speakers=2, detect_every_n=8, face_scale=1.5,
                 device='cpu', backend='auto'):
        self.num_speakers = num_speakers
        self.detect_every_n = detect_every_n
        self.face_scale = face_scale

        # Select backend
        if backend == 'auto':
            if _USE_MEDIAPIPE:
                self.backend = 'mediapipe'
            else:
                self.backend = 'opencv'
        else:
            self.backend = backend

        print(f"[StreamingFaceProcessor] Using backend: {self.backend}")

        # Backend-specific lazy-initialized models
        self._mp_face_detection = None
        self._mp_face_mesh = None
        self._cv_face_cascade = None
        self.device = device

        # State per speaker
        self.boxes = {i: None for i in range(num_speakers)}

        # For mediapipe backend: per-speaker face mesh results (used for mouth crop)
        self._mp_face_landmarks = {i: None for i in range(num_speakers)}

        # Frame counter
        self.frame_count = 0

        # Preprocessing pipeline (same as training val)
        self.preprocess = get_preprocessing_pipelines()["val"]

    # ---- Backend: mediapipe ----

    @property
    def mp_face_detection(self):
        if self._mp_face_detection is None:
            self._mp_face_detection = _mp_face_detection_cls(
                model_selection=0, min_detection_confidence=0.5
            )
        return self._mp_face_detection

    @property
    def mp_face_mesh(self):
        if self._mp_face_mesh is None:
            self._mp_face_mesh = _mp_face_mesh_cls(
                max_num_faces=self.num_speakers,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._mp_face_mesh

    def _detect_and_track_mediapipe(self, frame_bgr):
        """Run mediapipe face detection + mesh on one frame."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Face detection for bounding boxes
        det_results = self.mp_face_detection.process(frame_rgb)
        if det_results.detections:
            detected_boxes = []
            for det in det_results.detections[:self.num_speakers]:
                bbox = det.location_data.relative_bounding_box
                h, w = frame_bgr.shape[:2]
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))
                # Scale up by face_scale
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                sw = int(max(x2 - x1, y2 - y1) * self.face_scale)
                detected_boxes.append([
                    max(0, cx - sw/2), max(0, cy - sw/2),
                    min(w, cx + sw/2), min(h, cy + sw/2)
                ])
            self._match_boxes_to_speakers(detected_boxes)

        # Face mesh for landmarks (used for mouth cropping)
        mesh_results = self.mp_face_mesh.process(frame_rgb)
        if mesh_results.multi_face_landmarks:
            h, w = frame_bgr.shape[:2]
            # Match mesh faces to speaker boxes by proximity
            for face_lm in mesh_results.multi_face_landmarks[:self.num_speakers]:
                # Get face center from landmarks
                nose = face_lm.landmark[1]
                face_cx, face_cy = nose.x * w, nose.y * h

                # Find closest speaker box
                best_sid = 0
                best_dist = float('inf')
                for sid in range(self.num_speakers):
                    if self.boxes[sid] is not None:
                        bx = (self.boxes[sid][0] + self.boxes[sid][2]) / 2
                        by = (self.boxes[sid][1] + self.boxes[sid][3]) / 2
                        dist = (face_cx - bx)**2 + (face_cy - by)**2
                        if dist < best_dist:
                            best_dist = dist
                            best_sid = sid
                    else:
                        best_sid = sid
                        break
                self._mp_face_landmarks[best_sid] = face_lm

    def _crop_mouth_mediapipe(self, frame_bgr, sid):
        """Crop mouth region using mediapipe face mesh landmarks."""
        face_lm = self._mp_face_landmarks.get(sid)
        if face_lm is None:
            return None

        h, w = frame_bgr.shape[:2]

        # Get lip landmark positions
        lip_points = []
        for idx in _MP_LIP_LANDMARKS:
            lm = face_lm.landmark[idx]
            lip_points.append((int(lm.x * w), int(lm.y * h)))

        if not lip_points:
            return None

        # Bounding box of lip region with padding
        xs = [p[0] for p in lip_points]
        ys = [p[1] for p in lip_points]
        mouth_w = max(xs) - min(xs)
        mouth_h = max(ys) - min(ys)
        pad = max(mouth_w, mouth_h) * 0.3

        x1 = max(0, int(min(xs) - pad))
        y1 = max(0, int(min(ys) - pad))
        x2 = min(w, int(max(xs) + pad))
        y2 = min(h, int(max(ys) + pad))

        if x2 <= x1 or y2 <= y1:
            return None

        mouth_patch = frame_bgr[y1:y2, x1:x2]
        # Resize to 96x96 (preprocessing will CenterCrop to 88x88)
        mouth_patch = cv2.resize(mouth_patch, (96, 96))
        return mouth_patch

    # ---- Backend: opencv (Haar Cascade, always available) ----

    @property
    def cv_face_cascade(self):
        if self._cv_face_cascade is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._cv_face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._cv_face_cascade

    def _detect_faces_opencv(self, frame_bgr):
        """Run OpenCV Haar Cascade face detection. Returns head boxes."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cv_face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return []
        # Sort by face area (largest first) and take top num_speakers
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:self.num_speakers]
        # Convert (x, y, w, h) to [x1, y1, x2, y2] then scale to head box
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])
        boxes = face2head(boxes, self.face_scale)
        return boxes

    def _crop_mouth_opencv(self, frame_bgr, sid):
        """Crop mouth region from face box (lower portion heuristic).

        Since we don't have facial landmarks with OpenCV backend,
        we use the lower 40% of the face bounding box as the mouth region,
        with horizontal centering at 80% width.
        """
        box = self.boxes.get(sid)
        if box is None:
            return None

        h_frame, w_frame = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(b) for b in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        face_h = y2 - y1
        face_w = x2 - x1

        # Mouth region: lower 35% of face, centered at 80% width
        mouth_y1 = y1 + int(face_h * 0.65)
        mouth_y2 = y1 + int(face_h * 0.95)
        margin_x = int(face_w * 0.10)
        mouth_x1 = x1 + margin_x
        mouth_x2 = x2 - margin_x

        mouth_y1 = max(0, mouth_y1)
        mouth_y2 = min(h_frame, mouth_y2)
        mouth_x1 = max(0, mouth_x1)
        mouth_x2 = min(w_frame, mouth_x2)

        if mouth_x2 <= mouth_x1 or mouth_y2 <= mouth_y1:
            return None

        mouth_patch = frame_bgr[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        # Resize to 96x96 (preprocessing will CenterCrop to 88x88)
        mouth_patch = cv2.resize(mouth_patch, (96, 96))
        return mouth_patch

    # ---- Shared methods ----

    def _match_boxes_to_speakers(self, detected_boxes):
        """Match detected face boxes to existing speaker tracks using IoU."""
        if not detected_boxes:
            return

        matched_speakers = set()
        speaker_boxes = [None] * self.num_speakers

        for box in detected_boxes:
            iou_scores = []
            for sid in range(self.num_speakers):
                if sid in matched_speakers:
                    iou_scores.append(-1)
                elif self.boxes[sid] is not None:
                    iou_scores.append(bb_intersection_over_union(box, self.boxes[sid]))
                else:
                    iou_scores.append(0)
            if max(iou_scores) > 0:
                best_speaker = iou_scores.index(max(iou_scores))
                speaker_boxes[best_speaker] = box
                matched_speakers.add(best_speaker)

        for sid in range(self.num_speakers):
            if speaker_boxes[sid] is not None:
                self.boxes[sid] = speaker_boxes[sid]

    def process_frames(self, frames_bgr):
        """Process a chunk of video frames. Returns preprocessed mouth ROIs.

        Args:
            frames_bgr: list of numpy arrays (H, W, 3) in BGR format

        Returns:
            dict {speaker_id: np.array (num_frames, 88, 88)} preprocessed mouth ROIs
        """
        mouth_patches = {i: [] for i in range(self.num_speakers)}

        for frame_bgr in frames_bgr:
            should_detect = (self.frame_count % self.detect_every_n == 0
                             or self.boxes[0] is None)

            if self.backend == 'mediapipe':
                if should_detect:
                    self._detect_and_track_mediapipe(frame_bgr)

                for sid in range(self.num_speakers):
                    patch = self._crop_mouth_mediapipe(frame_bgr, sid)
                    if patch is not None:
                        mouth_patches[sid].append(patch)

            elif self.backend == 'opencv':
                if should_detect:
                    detected = self._detect_faces_opencv(frame_bgr)
                    if detected:
                        self._match_boxes_to_speakers(detected)

                for sid in range(self.num_speakers):
                    patch = self._crop_mouth_opencv(frame_bgr, sid)
                    if patch is not None:
                        mouth_patches[sid].append(patch)

            self.frame_count += 1

        # Convert to grayscale and preprocess
        result = {}
        for sid in range(self.num_speakers):
            if mouth_patches[sid]:
                gray_patches = convert_bgr2gray(np.array(mouth_patches[sid]))
                result[sid] = self.preprocess(gray_patches)
            else:
                result[sid] = np.array([])

        return result



# ============================================
# Streaming Separator (chunk-aware)
# ============================================
class StreamingSeparator:
    """Streaming audio separator with chunk-aware context.

    Matches training chunk mask behavior by feeding only the visible window
    (history + chunk + future) to the model without explicit chunk masking.
    Since the model was trained with restricted attention to the visible window,
    feeding only the visible window produces identical behavior.

    Args:
        model: Dolphin model instance
        device: torch device
        chunk_size_ms: chunk duration in ms (default 500)
        history_len_ms: history context in ms (default 200)
        future_len_ms: future lookahead in ms (default 100)
        crossfade_ms: crossfade duration between chunks in ms (default 20)
    """

    def __init__(self, model, device, chunk_size_ms=500,
                 history_len_ms=200, future_len_ms=100,
                 crossfade_ms=20):
        self.model = model
        self.device = device
        self.chunk_size_ms = chunk_size_ms
        self.history_len_ms = history_len_ms
        self.future_len_ms = future_len_ms
        self.crossfade_ms = crossfade_ms

        # Preprocessing for mouth ROI
        self.preprocess = get_preprocessing_pipelines()["val"]

        # State (initialized in initialize())
        self.sr = None
        self.chunk_size = None    # in samples
        self.history_len = None   # in samples
        self.future_len = None    # in samples
        self.crossfade_len = None # in samples

        self.history_audio = None
        self.future_audio = None
        self.output_buffers = {}

    def initialize(self, speaker_ids, sr):
        """Initialize streaming state.

        Args:
            speaker_ids: list of speaker IDs
            sr: audio sample rate
        """
        self.sr = sr
        self.chunk_size = int(sr * self.chunk_size_ms / 1000)
        self.history_len = int(sr * self.history_len_ms / 1000)
        self.future_len = int(sr * self.future_len_ms / 1000)
        self.crossfade_len = int(sr * self.crossfade_ms / 1000)

        # Audio buffers
        self.history_audio = None  # will be set after first chunk
        self.future_audio = None

        for sid in speaker_ids:
            self.output_buffers[sid] = []

        print(f"StreamingSeparator initialized:")
        print(f"  chunk_size={self.chunk_size} samples ({self.chunk_size_ms}ms)")
        print(f"  history_len={self.history_len} samples ({self.history_len_ms}ms)")
        print(f"  future_len={self.future_len} samples ({self.future_len_ms}ms)")
        print(f"  crossfade={self.crossfade_len} samples ({self.crossfade_ms}ms)")

    def process_chunk(self, audio_chunk, mouth_rois, speaker_ids,
                      future_audio_chunk=None):
        """Process one audio chunk with corresponding mouth ROIs.

        The model receives [history | chunk | future] as input, and we extract
        only the "chunk" portion of the output. This matches training chunk mask
        behavior without needing explicit masks.

        Args:
            audio_chunk: current chunk audio tensor (chunk_size samples,)
            mouth_rois: dict {speaker_id: np.array (T, 88, 88)}
            speaker_ids: list of speaker IDs
            future_audio_chunk: optional future audio for lookahead (future_len samples,)

        Returns:
            dict {speaker_id: separated audio tensor (chunk_size samples,)}
        """
        # 1. Assemble input audio: [history | chunk | future]
        audio_parts = []

        # History context
        history_available = 0
        if self.history_audio is not None and len(self.history_audio) > 0:
            hist = self.history_audio[-self.history_len:].to(self.device)
            history_available = len(hist)
            audio_parts.append(hist)

        # Current chunk
        audio_parts.append(audio_chunk.to(self.device))

        # Future lookahead
        if future_audio_chunk is not None and len(future_audio_chunk) > 0:
            fut = future_audio_chunk[:self.future_len].to(self.device)
            audio_parts.append(fut)

        audio_input = torch.cat(audio_parts)

        # 2. Prepare mouth ROIs: pad/truncate to match audio length
        samples_per_frame = self.sr / 25  # 640 samples per frame at 16kHz/25fps
        expected_frames = int(len(audio_input) / samples_per_frame)

        results = {}
        for sid in speaker_ids:
            mouth_data = mouth_rois[sid]
            if len(mouth_data) == 0:
                # No mouth data available yet, skip this speaker
                results[sid] = torch.zeros(self.chunk_size, device='cpu')
                continue

            # Adjust mouth ROI frames to match audio length
            if len(mouth_data) < expected_frames:
                # Repeat last frame to pad
                pad_count = expected_frames - len(mouth_data)
                mouth_data = np.concatenate([mouth_data, np.repeat(mouth_data[-1:], pad_count, axis=0)])
            elif len(mouth_data) > expected_frames:
                mouth_data = mouth_data[:expected_frames]

            # 3. Model inference (no chunk masking needed, visible window only)
            mouth_tensor = torch.from_numpy(mouth_data[None, None]).float().to(self.device)

            with torch.no_grad():
                est = self.model(audio_input[None], mouth_tensor)

            # 4. Extract the "chunk" portion from output
            # Output corresponds to the audio_input length
            # The chunk starts after history_available samples
            output = est[0, 0].cpu()  # (audio_input_len,)

            if history_available + self.chunk_size <= len(output):
                chunk_output = output[history_available:history_available + self.chunk_size]
            else:
                # Output might be shorter due to encoder stride rounding
                chunk_output = output[history_available:]
                if len(chunk_output) < self.chunk_size:
                    chunk_output = torch.cat([
                        chunk_output,
                        torch.zeros(self.chunk_size - len(chunk_output))
                    ])

            results[sid] = chunk_output

        # 5. Update history buffer
        self.history_audio = audio_chunk.clone()

        # 6. Crossfade with previous output
        for sid in speaker_ids:
            if len(self.output_buffers[sid]) > 0 and self.crossfade_len > 0:
                prev = self.output_buffers[sid][-1]
                curr = results[sid]
                cf_len = min(self.crossfade_len, len(prev), len(curr))
                if cf_len > 0:
                    fade_out = torch.linspace(1, 0, cf_len)
                    fade_in = torch.linspace(0, 1, cf_len)
                    prev_tail = prev[-cf_len:] * fade_out
                    curr_head = curr[:cf_len] * fade_in
                    crossfaded = prev_tail + curr_head
                    merged = torch.cat([prev[:-cf_len], crossfaded, curr[cf_len:]])
                    self.output_buffers[sid][-1] = merged
                else:
                    self.output_buffers[sid].append(results[sid])
            else:
                self.output_buffers[sid].append(results[sid])

        return results

    def finalize(self, speaker_ids):
        """Concatenate all chunk outputs and return final results."""
        final_results = {}
        for sid in speaker_ids:
            if self.output_buffers[sid]:
                final_results[sid] = torch.cat(self.output_buffers[sid])
            else:
                final_results[sid] = torch.tensor([])
        return final_results


# ============================================
# Main streaming pipeline
# ============================================
def process_video_streaming(input_file, output_path, number_of_speakers=2,
                            detect_every_N_frame=8, scalar_face_detection=1.5,
                            chunk_duration_ms=500,
                            history_len_ms=200, future_len_ms=100,
                            crossfade_ms=20,
                            cuda_device=None, timing_stats=None):
    """Streaming video processing pipeline.

    Processes video and audio in sync, frame-by-frame and chunk-by-chunk,
    without loading the entire video into memory.

    Args:
        chunk_duration_ms: chunk duration in ms
        history_len_ms: history context in ms (matching training chunk mask)
        future_len_ms: future lookahead in ms (matching training chunk mask)
        crossfade_ms: crossfade between chunks in ms
    """
    if timing_stats is None:
        timing_stats = ModuleTimingStats()

    use_cuda = cuda_device is not None and cuda_device >= 0 and torch.cuda.is_available()
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    os.makedirs(output_path, exist_ok=True)
    device = torch.device('cuda' if use_cuda else 'cpu')

    sr = 16000
    chunk_samples = int(sr * chunk_duration_ms / 1000)
    history_samples = int(sr * history_len_ms / 1000)
    future_samples = int(sr * future_len_ms / 1000)

    # ===== Step 1: Extract audio from video =====
    with Timer("Audio extraction") as t:
        audio_output = os.path.join(output_path, 'audio.wav')
        extract_audio(input_file, audio_output, sample_rate=sr)
    timing_stats.record("Audio extraction", t.elapsed)

    # ===== Step 2: Load model =====
    with Timer("Model loading") as t:
        audiomodel = Dolphin.from_pretrained("JusperLee/Dolphin",
            cache_dir="/data/megastore/Projects/DuJing/code/Dolphin/ckpt")
        if use_cuda:
            audiomodel.cuda()
        else:
            audiomodel.cpu()
        audiomodel.eval()
    timing_stats.record("Model loading", t.elapsed)

    # ===== Step 3: Initialize streaming components =====
    speaker_ids = list(range(number_of_speakers))

    face_processor = StreamingFaceProcessor(
        num_speakers=number_of_speakers,
        detect_every_n=detect_every_N_frame,
        face_scale=scalar_face_detection,
        device='cuda' if use_cuda else 'cpu'
    )

    separator = StreamingSeparator(
        model=audiomodel,
        device=device,
        chunk_size_ms=chunk_duration_ms,
        history_len_ms=history_len_ms,
        future_len_ms=future_len_ms,
        crossfade_ms=crossfade_ms
    )
    separator.initialize(speaker_ids, sr)

    # ===== Step 4: Load audio =====
    mix, sr = torchaudio.load(audio_output)
    mix = mix.mean(dim=0)  # mono
    total_samples = len(mix)
    total_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    print(f"\nStreaming inference config:")
    print(f"  Total audio: {total_samples/sr:.2f}s")
    print(f"  Chunk: {chunk_samples} samples ({chunk_duration_ms}ms)")
    print(f"  History: {history_samples} samples ({history_len_ms}ms)")
    print(f"  Future: {future_samples} samples ({future_len_ms}ms)")
    print(f"  Total chunks: {total_chunks}\n")

    # ===== Step 5: Open video and process chunks =====
    sep_timing = SeparatorTimingStats()
    total_infer_time = 0
    chunk_count = 0

    with Timer("Streaming inference") as t_stream:
        cap = cv2.VideoCapture(input_file)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        with torch.no_grad():
            pos = 0
            chunk_idx = 0

            while pos < total_samples:
                # Determine audio chunk boundaries
                chunk_end = min(pos + chunk_samples, total_samples)
                audio_chunk = mix[pos:chunk_end].clone()

                # Pad chunk if too short
                if len(audio_chunk) < chunk_samples:
                    audio_chunk = torch.cat([
                        audio_chunk,
                        torch.zeros(chunk_samples - len(audio_chunk))
                    ])

                # Get future audio chunk for lookahead
                future_audio = None
                if future_samples > 0 and chunk_end < total_samples:
                    future_end = min(chunk_end + future_samples, total_samples)
                    future_audio = mix[chunk_end:future_end].clone()
                    if len(future_audio) < future_samples:
                        future_audio = torch.cat([
                            future_audio,
                            torch.zeros(future_samples - len(future_audio))
                        ])

                # Read corresponding video frames
                frame_end = int(chunk_end / sr * video_fps)
                # History frames
                hist_frame_start = max(0, int((pos - history_samples) / sr * video_fps))
                # Future frames
                future_frame_end = int(min(total_samples, chunk_end + future_samples) / sr * video_fps) if future_samples > 0 else frame_end

                # Read frames from video
                frames_to_read = list(range(hist_frame_start, future_frame_end))
                frames_bgr = []
                for fidx in frames_to_read:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                    ret, frame = cap.read()
                    if ret:
                        frames_bgr.append(frame)
                    elif frames_bgr:
                        # Repeat last frame if video ends
                        frames_bgr.append(frames_bgr[-1])

                # Process face tracking and mouth cropping
                with Timer("Face+Mouth") as t_face:
                    mouth_rois = face_processor.process_frames(frames_bgr)
                face_time = t_face.elapsed

                # If mouth ROIs don't have enough frames, pad with zeros
                for sid in speaker_ids:
                    if len(mouth_rois[sid]) == 0:
                        # Create dummy mouth ROIs
                        needed_frames = len(frames_bgr) if frames_bgr else 1
                        mouth_rois[sid] = np.zeros((needed_frames, 88, 88), dtype=np.float32)
                        mouth_rois[sid] = face_processor.preprocess(mouth_rois[sid])

                # Model inference
                infer_start = time.perf_counter()
                separator.process_chunk(
                    audio_chunk, mouth_rois, speaker_ids,
                    future_audio_chunk=future_audio
                )
                infer_time = time.perf_counter() - infer_start
                total_infer_time += infer_time
                chunk_count += 1

                audio_duration_sec = len(audio_chunk) / sr
                sep_timing.record_chunk(infer_time, audio_duration_sec)

                pos += chunk_samples
                chunk_idx += 1

                print(f"\rChunk {chunk_idx}/{total_chunks} | "
                      f"Progress: {pos/total_samples*100:.1f}% | "
                      f"Face: {face_time*1000:.0f}ms | "
                      f"Infer: {infer_time*1000:.0f}ms", end='')

        cap.release()

    print(f"\n\nStreaming inference complete!")
    print(f"Avg chunk inference: {total_infer_time/chunk_count*1000:.2f}ms")
    timing_stats.record("Streaming inference total", t_stream.elapsed)
    timing_stats.record("Model inference total", total_infer_time)

    # ===== Step 6: Finalize and save =====
    with Timer("Output saving") as t:
        final_outputs = separator.finalize(speaker_ids)
        for sid in speaker_ids:
            # Trim output to match original audio length
            output = final_outputs[sid][:total_samples]
            wav_path = os.path.join(output_path, f"speaker{sid+1}_est.wav")
            torchaudio.save(wav_path, output.unsqueeze(0), sr)
            print(f"Speaker {sid+1} saved to: {wav_path}")
    timing_stats.record("Output saving", t.elapsed)

    sep_timing.summary()
    output_files = [os.path.join(output_path, f"speaker{sid+1}_est.wav") for sid in speaker_ids]
    return output_files, timing_stats


# ============================================
# Video/Audio helper functions (unchanged)
# ============================================
def extract_audio(video_file, audio_output_file, sample_rate=16000):
    """Extract audio from video using torchaudio."""
    from moviepy import VideoFileClip
    video = VideoFileClip(video_file)
    audio = video.audio
    if audio is not None:
        audio.write_audiofile(audio_output_file, fps=sample_rate, nbytes=2, codec='pcm_s16le')
        audio.close()
    video.close()


def merge_video_audio(video_file, audio_file, output_file):
    """Merge video and audio."""
    from moviepy import VideoFileClip, AudioFileClip
    video = VideoFileClip(video_file)
    audio = AudioFileClip(audio_file)

    set_audio_fn = getattr(video, "set_audio", None)
    if callable(set_audio_fn):
        final_video = set_audio_fn(audio)
    else:
        with_audio_fn = getattr(video, "with_audio", None)
        if not callable(with_audio_fn):
            video.close()
            audio.close()
            raise AttributeError("VideoFileClip lacks both set_audio and with_audio methods")
        final_video = with_audio_fn(audio)

    final_video.write_videofile(output_file, codec='libx264', audio_codec='aac',
                                temp_audiofile='temp-audio.m4a', remove_temp=True)
    video.close()
    audio.close()
    final_video.close()


# ============================================
# Entry point
# ============================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streaming video speaker separation')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input video file path')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory path')
    parser.add_argument('--speakers', '-s', type=int, default=2,
                        help='Number of speakers (default: 2)')
    parser.add_argument('--chunk-ms', type=int, default=500,
                        help='Chunk duration in ms (default: 500)')
    parser.add_argument('--history-ms', type=int, default=500,
                        help='History context in ms (default: 500)')
    parser.add_argument('--future-ms', type=int, default=100,
                        help='Future lookahead in ms (default: 100)')
    parser.add_argument('--crossfade-ms', type=int, default=20,
                        help='Crossfade between chunks in ms (default: 20)')
    parser.add_argument('--detect-every-n', type=int, default=8,
                        help='Face detection every N frames (default: 8)')
    parser.add_argument('--face-scale', type=float, default=1.5,
                        help='Face detection box scale factor (default: 1.5)')
    parser.add_argument('--cuda-device', type=int, default=0,
                        help='CUDA device ID (default: 0, -1 for CPU)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode')
    parser.add_argument('--timing', action='store_true',
                        help='Print detailed timing stats')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' does not exist")
        exit(1)

    if args.output is None:
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(os.path.dirname(args.input), input_basename + "_output_streaming")

    if args.cpu:
        cuda_device = None
    else:
        cuda_device = args.cuda_device if args.cuda_device >= 0 else None

    print("="*60)
    print("Streaming Video Speaker Separation")
    print("="*60)
    print(f"Input:     {args.input}")
    print(f"Output:    {args.output}")
    print(f"Speakers:  {args.speakers}")
    print(f"Chunk:     {args.chunk_ms}ms")
    print(f"History:   {args.history_ms}ms")
    print(f"Future:    {args.future_ms}ms")
    print(f"Device:    {'CUDA:' + str(cuda_device) if cuda_device is not None else 'CPU'}")
    print("="*60 + "\n")

    timing_stats = ModuleTimingStats() if args.timing else None

    start_time = time.perf_counter()
    output_files, timing_stats = process_video_streaming(
        input_file=args.input,
        output_path=args.output,
        number_of_speakers=args.speakers,
        detect_every_N_frame=args.detect_every_n,
        scalar_face_detection=args.face_scale,
        chunk_duration_ms=args.chunk_ms,
        history_len_ms=args.history_ms,
        future_len_ms=args.future_ms,
        crossfade_ms=args.crossfade_ms,
        cuda_device=cuda_device,
        timing_stats=timing_stats
    )
    total_time = time.perf_counter() - start_time

    if timing_stats:
        timing_stats.record("Total processing time", total_time)
        timing_stats.summary()

    print("\nDone!")
    for i, f in enumerate(output_files):
        print(f"  Speaker {i+1}: {f}")
