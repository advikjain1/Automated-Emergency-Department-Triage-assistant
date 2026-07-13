This folder contains project code files
Audio Sensor code:
import numpy as np
import sounddevice as sd
import time
from collections import deque

SAMPLE_RATE = 44100
CHUNK = 1024
CONSECUTIVE_FRAMES_REQUIRED = 3
NOISE_FLOOR_WINDOW = 100          # frames used for rolling adaptive baseline (~2.3s worth)
ONSET_RATIO_REQUIRED = 3.0        # frame must be this many x the rolling noise floor
FLATNESS_MIN_FOR_COUGH = 0.15     # coughs are noisy/broadband -> higher spectral flatness
PRINT_INTERVAL_SECONDS = 1.0      # how often you SEE output, not how often it analyzes


def compute_features(audio_chunk):
    """Extract volume, dominant frequency, spectral flatness, and zero-crossing rate."""
    samples = (audio_chunk[:, 0] if audio_chunk.ndim > 1 else audio_chunk).astype(np.float64)

    rms = np.sqrt(np.mean(samples ** 2))
    vol_peak = rms * 1000

    windowed = samples * np.hanning(len(samples))
    fft_vals = np.fft.rfft(windowed)
    fft_freqs = np.fft.rfftfreq(len(windowed), d=1.0 / SAMPLE_RATE)
    magnitude = np.abs(fft_vals) + 1e-12  # avoid log(0)

    freq = 0.0 if magnitude.max() < 1e-6 else fft_freqs[np.argmax(magnitude)]

    # Spectral flatness: geometric_mean / arithmetic_mean of the spectrum.
    # Near 1.0 = noise-like/broadband (cough, hiss). Near 0 = tonal/narrowband (hum, whistle, steady AC).
    log_mag = np.log(magnitude)
    geo_mean = np.exp(np.mean(log_mag))
    arith_mean = np.mean(magnitude)
    flatness = geo_mean / arith_mean

    # Zero-crossing rate: how often the signal flips sign per sample.
    # High for noisy/turbulent sounds like coughs, breath, fricatives; lower for smooth tones/hums.
    zcr = np.mean(np.abs(np.diff(np.sign(samples)))) / 2

    return vol_peak, freq, flatness, zcr


def run_audio_sensor():
    print("Mic Active. Initializing Audio Triage Stream Monitoring Engine...")
    print("Press Ctrl + C in the terminal window to halt.")

    noise_floor_history = deque(maxlen=NOISE_FLOOR_WINDOW)
    consecutive_hits = 0
    recent_freqs = deque(maxlen=CONSECUTIVE_FRAMES_REQUIRED)
    last_print_time = 0.0

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK) as stream:
            print("Calibrating ambient noise for 3s... stay quiet.")
            for _ in range(int(3 * SAMPLE_RATE / CHUNK)):
                audio_chunk, _ = stream.read(CHUNK)
                vol_peak, _, _, _ = compute_features(audio_chunk)
                noise_floor_history.append(vol_peak)
            print(f"Calibration done. Initial noise floor: {np.mean(noise_floor_history):.1f}\n")

            while True:
                audio_chunk, _ = stream.read(CHUNK)
                vol_peak, freq, flatness, zcr = compute_features(audio_chunk)

                # Rolling adaptive noise floor — median is robust to occasional loud spikes
                noise_floor = np.median(noise_floor_history)
                onset_ratio = vol_peak / max(noise_floor, 1e-6)

                is_sudden_and_loud = onset_ratio > ONSET_RATIO_REQUIRED
                is_broadband = flatness > FLATNESS_MIN_FOR_COUGH

                recent_freqs.append(freq)

                if is_sudden_and_loud and is_broadband:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0
                    # Only let quiet/steady frames update the noise floor
                    noise_floor_history.append(vol_peak)

                status = "Normal Ambient Sound"
                if consecutive_hits >= CONSECUTIVE_FRAMES_REQUIRED:
                    avg_freq = np.mean(recent_freqs)
                    if 600.0 <= avg_freq <= 1000.0:
                        status = "Wheezing / Stridor Detected"
                    elif 100.0 <= avg_freq <= 800.0:
                        status = "Severe Coughing Fit Detected"
                    else:
                        status = "Sudden Loud Noise (Unclassified)"
                elif onset_ratio > ONSET_RATIO_REQUIRED and not is_broadband:
                    status = "Steady Background Noise / Continuous Audio"

                # A detected event always prints immediately, regardless of pacing,
                # so you never miss the moment it actually fires.
                now = time.time()
                should_print = (now - last_print_time >= PRINT_INTERVAL_SECONDS) or (status != "Normal Ambient Sound")

                if should_print:
                    print(f"Mic Active | Vol: {vol_peak:.1f} | Floor: {noise_floor:.1f} | "
                          f"Freq: {freq:.1f} Hz | Hits: {consecutive_hits} | Status: {status}")
                    last_print_time = now

    except KeyboardInterrupt:
        print("\nAudio monitoring stream stopped cleanly.")


if __name__ == "__main__":
    run_audio_sensor()
Video Sensor code :
import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# --- Face landmark indices (MediaPipe Face Mesh topology) ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_BROW, LEFT_EYE_TOP = 105, 159
RIGHT_BROW, RIGHT_EYE_TOP = 334, 386
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
FACE_TOP, FACE_BOTTOM = 10, 152

# --- Pose landmark indices (MediaPipe Pose) ---
LM = mp_pose.PoseLandmark
CALIBRATION_SECONDS = 2.0
GAIT_WINDOW = 30
TENSION_SMOOTHING = 15

# --- Performance: process a downscaled copy through MediaPipe, draw on the full-res frame.
# Running FaceMesh(refine_landmarks=True) + Pose at full webcam resolution every frame is
# expensive and can silently tank FPS, which is the main reason timing-based detection felt
# inconsistent/buggy -- "20 frames" means something different at 10fps vs 30fps. Landmarks are
# normalized (0-1), so they still map correctly onto the full-res frame for drawing/scoring.
PROCESSING_WIDTH = 480

# --- Strain signal gating ---
THRESHOLD_STD_MULTIPLIER = 3.0
MIN_SIGNAL_THRESHOLD = 0.03
MIN_SIGNALS_FOR_GENUINE_STRAIN = 2
STRONG_SIGNAL_MULTIPLIER = 1.8
SATURATION_MULTIPLIER = 1.8

# --- Smile detection ---
SMILE_CORNER_LIFT_THRESHOLD = 0.015
SMILE_WIDTH_INCREASE_THRESHOLD = 0.08

# --- Head-pose reliability gate ---
HEAD_POSE_TOLERANCE = 0.30

# --- Slouch / Guard ---
SLOUCH_RATIO_THRESHOLD = 0.75
GUARD_PROXIMITY_RATIO = 0.5
# Loosened slightly: hugging your own stomach naturally causes some self-occlusion of the
# wrist/forearm, which lowers MediaPipe's confidence on those landmarks. 0.5 was rejecting
# valid-but-partially-occluded wrist data during exactly the pose we want to detect.
LANDMARK_VISIBILITY_THRESHOLD = 0.4

WRIST_CLASP_MAX_DIST_RATIO = 0.35
SINGLE_WRIST_HORIZONTAL_RATIO = 0.65
SINGLE_WRIST_VERTICAL_RATIO = 2.0

# --- All hold/decay behavior is now TIME-based (seconds), not frame-count-based. This is the
# core fix for inconsistent detection: frame counts only mean a fixed duration at a fixed FPS,
# and FPS here is neither fixed nor guaranteed. Seconds behave the same regardless of FPS. ---
GUARD_SECONDS_REQUIRED = 0.5
GUARD_DECAY_PER_SECOND = 1.5       # progress lost per second while NOT guarding
STRAIN_BUILD_SECONDS_REQUIRED = 0.25
STRAIN_DECAY_PER_SECOND = 2.5
FACIAL_STRESS_RELEASE_PER_SECOND = 0.6   # fraction of max stress drained per second when not reinforced
PROLONGED_SLOUCH_SECONDS = 3.0
SLOUCH_DECAY_PER_SECOND = 4.0

# --- Fusion scoring weights ---
MAX_STRAIN_SCORE_CONTRIBUTION = 6.0
GUARD_SCORE = 3.0
SLOUCH_SCORE = 1.5
PROLONGED_SLOUCH_SCORE = 2.0
LIMP_SCORE = 3.0
STRAIN_GUARD_COMBO_BONUS = 2.0
COMBO_STRAIN_THRESHOLD = 0.3


def _dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def eye_aspect_ratio(lm, eye_idx, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = pts
    vertical = (_dist(p2, p6) + _dist(p3, p5)) / 2.0
    horizontal = _dist(p1, p4)
    return vertical / horizontal if horizontal > 1e-6 else 0.0


def facial_raw_metrics(face_landmarks, w, h):
    lm = face_landmarks.landmark
    ear = (eye_aspect_ratio(lm, LEFT_EYE, w, h) + eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2

    face_height = max(abs(lm[FACE_BOTTOM].y * h - lm[FACE_TOP].y * h), 1e-6)
    brow_dist = (abs(lm[LEFT_BROW].y * h - lm[LEFT_EYE_TOP].y * h) +
                 abs(lm[RIGHT_BROW].y * h - lm[RIGHT_EYE_TOP].y * h)) / 2 / face_height

    mouth_h = _dist((lm[MOUTH_TOP].x * w, lm[MOUTH_TOP].y * h),
                     (lm[MOUTH_BOTTOM].x * w, lm[MOUTH_BOTTOM].y * h))
    mouth_openness = mouth_h / face_height

    mouth_width = _dist((lm[MOUTH_LEFT].x * w, lm[MOUTH_LEFT].y * h),
                         (lm[MOUTH_RIGHT].x * w, lm[MOUTH_RIGHT].y * h)) / face_height

    mouth_center_y = (lm[MOUTH_TOP].y * h + lm[MOUTH_BOTTOM].y * h) / 2
    corner_lift = ((mouth_center_y - lm[MOUTH_LEFT].y * h) +
                    (mouth_center_y - lm[MOUTH_RIGHT].y * h)) / 2 / face_height

    left_eye_center = np.mean([(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE], axis=0)
    right_eye_center = np.mean([(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE], axis=0)
    eye_distance = max(_dist(left_eye_center, right_eye_center), 1e-6)
    face_shape_ratio = face_height / eye_distance

    return ear, brow_dist, mouth_openness, mouth_width, corner_lift, face_shape_ratio


def compute_facial_stress_components(ear, brow_dist, mouth_openness, baseline):
    base_ear, base_brow, base_mouth_openness = baseline
    squint = max(0.0, (base_ear - ear) / max(base_ear, 1e-6))
    furrow = max(0.0, (base_brow - brow_dist) / max(base_brow, 1e-6))
    mouth_dev = abs(mouth_openness - base_mouth_openness)
    return squint, furrow, mouth_dev


def detect_smile(mouth_width, corner_lift, baseline_width, baseline_corner_lift):
    width_increase = mouth_width - baseline_width
    lift_increase = corner_lift - baseline_corner_lift
    return width_increase > SMILE_WIDTH_INCREASE_THRESHOLD and lift_increase > SMILE_CORNER_LIFT_THRESHOLD


def _saturate(raw_value, threshold):
    if threshold <= 1e-6:
        return 0.0
    cap = threshold * SATURATION_MULTIPLIER
    return float(np.clip(raw_value / cap, 0.0, 1.0))


def run_vision_sensor():
    print("Initializing Multi-Modal Posture & Face Fusion Tracking Engine (MediaPipe)...")

    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    tension_history = deque(maxlen=TENSION_SMOOTHING)
    left_ankle_y_hist = deque(maxlen=GAIT_WINDOW)
    right_ankle_y_hist = deque(maxlen=GAIT_WINDOW)

    # Time-based hold state (seconds accumulated, not frame counts)
    guard_seconds = 0.0
    strain_seconds = 0.0
    slouch_seconds = 0.0
    facial_stress_smoothed = 0.0

    baseline = None
    baseline_torso = None
    baseline_wrist_dist = None
    baseline_mouth_width = None
    baseline_corner_lift = None
    baseline_face_shape_ratio = None
    squint_thresh = furrow_thresh = mouth_thresh = None
    calib_ear, calib_brow, calib_mouth = [], [], []
    calib_mouth_width, calib_corner_lift, calib_face_shape = [], [], []
    calib_torso, calib_wrist_dist = [], []

    calibration_start = None
    calibration_done_face = False
    calibration_done_pose = False

    print("Calibrating neutral face and posture for ~2s.")
    print("IMPORTANT: hold the SAME head angle/distance you'll use during testing.")
    print("Sit in your normal resting position (hands NOT near your stomach), relax your face, don't smile...")

    last_time = time.time()
    fps_smoothed = 30.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = max(0.0, min(now - last_time, 0.2))  # clamp dt to avoid huge jumps after a stall
        last_time = now
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * (1.0 / dt if dt > 1e-3 else fps_smoothed)

        try:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Downscaled copy for MediaPipe processing (speed), landmarks still map to full-res frame.
            proc_scale = PROCESSING_WIDTH / w if w > PROCESSING_WIDTH else 1.0
            if proc_scale < 1.0:
                proc_frame = cv2.resize(frame, (int(w * proc_scale), int(h * proc_scale)))
            else:
                proc_frame = frame
            proc_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

            face_results = face_mesh.process(proc_rgb)
            pose_results = pose.process(proc_rgb)

            if calibration_start is None:
                calibration_start = now
            calibrating = (now - calibration_start) < CALIBRATION_SECONDS

            facial_stress = 0.0
            is_limping = False
            is_guarding = False
            is_slouching = False
            is_smiling = False
            sustained_genuine_strain = False
            head_pose_reliable = True
            torso_span = None
            squint = furrow = mouth_dev = 0.0

            # ---- FACE ----
            if face_results.multi_face_landmarks:
                face_lms = face_results.multi_face_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, face_lms, mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=1))

                ear, brow_dist, mouth_openness, mouth_width, corner_lift, face_shape_ratio = \
                    facial_raw_metrics(face_lms, w, h)

                if calibrating:
                    calib_ear.append(ear)
                    calib_brow.append(brow_dist)
                    calib_mouth.append(mouth_openness)
                    calib_mouth_width.append(mouth_width)
                    calib_corner_lift.append(corner_lift)
                    calib_face_shape.append(face_shape_ratio)
                    cv2.putText(frame, "Calibrating... HOLD STILL, DON'T SMILE", (30, 50),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
                else:
                    if not calibration_done_face and len(calib_ear) > 0:
                        baseline = (np.mean(calib_ear), np.mean(calib_brow), np.mean(calib_mouth))
                        baseline_mouth_width = np.mean(calib_mouth_width)
                        baseline_corner_lift = np.mean(calib_corner_lift)
                        baseline_face_shape_ratio = np.mean(calib_face_shape)

                        ear_std = np.std(calib_ear) / max(baseline[0], 1e-6)
                        brow_std = np.std(calib_brow) / max(baseline[1], 1e-6)
                        mouth_std = np.std(calib_mouth)

                        squint_thresh = max(ear_std * THRESHOLD_STD_MULTIPLIER, MIN_SIGNAL_THRESHOLD)
                        furrow_thresh = max(brow_std * THRESHOLD_STD_MULTIPLIER, MIN_SIGNAL_THRESHOLD)
                        mouth_thresh = max(mouth_std * THRESHOLD_STD_MULTIPLIER, MIN_SIGNAL_THRESHOLD)
                        calibration_done_face = True
                        print(f"[Vision] Face calibration done. Thresholds -> squint:{squint_thresh:.3f} "
                              f"furrow:{furrow_thresh:.3f} mouth:{mouth_thresh:.3f}")

                    if baseline is not None:
                        shape_deviation = abs(face_shape_ratio - baseline_face_shape_ratio) / max(baseline_face_shape_ratio, 1e-6)
                        head_pose_reliable = shape_deviation <= HEAD_POSE_TOLERANCE

                        if head_pose_reliable:
                            is_smiling = detect_smile(mouth_width, corner_lift, baseline_mouth_width, baseline_corner_lift)
                            squint, furrow, mouth_dev = compute_facial_stress_components(ear, brow_dist, mouth_openness, baseline)

                            signals_active = sum([
                                squint > squint_thresh,
                                furrow > furrow_thresh,
                                mouth_dev > mouth_thresh,
                            ])
                            strong_single_signal = (
                                squint > squint_thresh * STRONG_SIGNAL_MULTIPLIER or
                                furrow > furrow_thresh * STRONG_SIGNAL_MULTIPLIER or
                                mouth_dev > mouth_thresh * STRONG_SIGNAL_MULTIPLIER
                            )
                            frame_has_genuine_strain = (
                                (signals_active >= MIN_SIGNALS_FOR_GENUINE_STRAIN) or strong_single_signal
                            ) and not is_smiling

                            if frame_has_genuine_strain:
                                strain_seconds += dt
                            else:
                                strain_seconds = max(0.0, strain_seconds - STRAIN_DECAY_PER_SECOND * dt)
                            sustained_genuine_strain = strain_seconds >= STRAIN_BUILD_SECONDS_REQUIRED

                            if is_smiling:
                                facial_stress = 0.0
                            else:
                                norm_squint = _saturate(squint, squint_thresh)
                                norm_furrow = _saturate(furrow, furrow_thresh)
                                norm_mouth = _saturate(mouth_dev, mouth_thresh)
                                raw_blend = 0.4 * norm_squint + 0.35 * norm_furrow + 0.25 * norm_mouth
                                facial_stress = float(np.clip(raw_blend, 0.0, 1.0))
                        # else: head angle too different from calibration -- skip facial scoring
                        # this frame, strain_seconds untouched (no reliable data either way).

            # Smooth facial_stress with time-scaled release (fast rise, slow fall).
            if facial_stress > facial_stress_smoothed:
                facial_stress_smoothed = facial_stress
            else:
                facial_stress_smoothed = max(
                    facial_stress, facial_stress_smoothed - FACIAL_STRESS_RELEASE_PER_SECOND * dt
                )

            # ---- POSE ----
            if pose_results.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2))

                l_sh, r_sh = lm[LM.LEFT_SHOULDER.value], lm[LM.RIGHT_SHOULDER.value]
                l_hip, r_hip = lm[LM.LEFT_HIP.value], lm[LM.RIGHT_HIP.value]
                l_wrist, r_wrist = lm[LM.LEFT_WRIST.value], lm[LM.RIGHT_WRIST.value]
                l_ankle, r_ankle = lm[LM.LEFT_ANKLE.value], lm[LM.RIGHT_ANKLE.value]

                hips_visible = (l_hip.visibility > LANDMARK_VISIBILITY_THRESHOLD and
                                 r_hip.visibility > LANDMARK_VISIBILITY_THRESHOLD)
                wrists_visible = (l_wrist.visibility > LANDMARK_VISIBILITY_THRESHOLD or
                                   r_wrist.visibility > LANDMARK_VISIBILITY_THRESHOLD)
                both_wrists_visible = (l_wrist.visibility > LANDMARK_VISIBILITY_THRESHOLD and
                                        r_wrist.visibility > LANDMARK_VISIBILITY_THRESHOLD)
                ankles_visible = (l_ankle.visibility > LANDMARK_VISIBILITY_THRESHOLD and
                                   r_ankle.visibility > LANDMARK_VISIBILITY_THRESHOLD)

                shoulder_mid_y = (l_sh.y + r_sh.y) / 2
                shoulder_width = max(_dist((l_sh.x, l_sh.y), (r_sh.x, r_sh.y)), 1e-6)
                hip_mid_y = (l_hip.y + r_hip.y) / 2
                torso_span = max(abs(hip_mid_y - shoulder_mid_y), 1e-6)

                if calibrating:
                    calib_torso.append(torso_span)
                elif not calibration_done_pose and len(calib_torso) > 0:
                    baseline_torso = np.mean(calib_torso)
                    calibration_done_pose = True
                    print(f"[Vision] Posture calibration done. Baseline torso span={baseline_torso:.3f}")

                if baseline_torso is not None:
                    is_slouching = torso_span < baseline_torso * SLOUCH_RATIO_THRESHOLD

                if is_slouching:
                    slouch_seconds += dt
                else:
                    slouch_seconds = max(0.0, slouch_seconds - SLOUCH_DECAY_PER_SECOND * dt)

                hip_based_guard = False
                if hips_visible and wrists_visible:
                    abdomen = ((l_hip.x + r_hip.x) / 2, hip_mid_y - torso_span * 0.15)
                    current_min_wrist_dist = min(_dist((l_wrist.x, l_wrist.y), abdomen),
                                                  _dist((r_wrist.x, r_wrist.y), abdomen))
                    if calibrating:
                        calib_wrist_dist.append(current_min_wrist_dist)
                    elif baseline_wrist_dist is None and len(calib_wrist_dist) > 0:
                        baseline_wrist_dist = np.mean(calib_wrist_dist)
                        print(f"[Vision] Guard calibration done. Baseline dist={baseline_wrist_dist:.3f}")
                    if baseline_wrist_dist is not None:
                        hip_based_guard = current_min_wrist_dist < baseline_wrist_dist * GUARD_PROXIMITY_RATIO

                clasp_guard = False
                if both_wrists_visible:
                    wrist_gap = _dist((l_wrist.x, l_wrist.y), (r_wrist.x, r_wrist.y))
                    torso_center_x = (l_sh.x + r_sh.x) / 2
                    wrists_center_x = (l_wrist.x + r_wrist.x) / 2
                    wrists_near_center = abs(wrists_center_x - torso_center_x) < shoulder_width * 0.6
                    clasp_guard = (wrist_gap < shoulder_width * WRIST_CLASP_MAX_DIST_RATIO) and wrists_near_center

                single_wrist_hug = False
                torso_center_x = (l_sh.x + r_sh.x) / 2
                for wrist, vis in ((l_wrist, l_wrist.visibility), (r_wrist, r_wrist.visibility)):
                    if vis <= LANDMARK_VISIBILITY_THRESHOLD:
                        continue
                    horiz_ok = abs(wrist.x - torso_center_x) < shoulder_width * SINGLE_WRIST_HORIZONTAL_RATIO
                    vert_ok = (wrist.y > shoulder_mid_y and
                               wrist.y < shoulder_mid_y + shoulder_width * SINGLE_WRIST_VERTICAL_RATIO)
                    if horiz_ok and vert_ok:
                        single_wrist_hug = True
                        break

                wrist_near = hip_based_guard or clasp_guard or single_wrist_hug

                if wrist_near:
                    guard_seconds += dt
                else:
                    guard_seconds = max(0.0, guard_seconds - GUARD_DECAY_PER_SECOND * dt)
                is_guarding = guard_seconds >= GUARD_SECONDS_REQUIRED

                if ankles_visible:
                    left_ankle_y_hist.append(l_ankle.y)
                    right_ankle_y_hist.append(r_ankle.y)
                    if len(left_ankle_y_hist) == GAIT_WINDOW:
                        l_range = max(left_ankle_y_hist) - min(left_ankle_y_hist)
                        r_range = max(right_ankle_y_hist) - min(right_ankle_y_hist)
                        total_motion = l_range + r_range
                        if total_motion > 0.05:
                            asymmetry = abs(l_range - r_range) / max(total_motion, 1e-6)
                            is_limping = asymmetry > 0.35
            else:
                slouch_seconds = max(0.0, slouch_seconds - SLOUCH_DECAY_PER_SECOND * dt)
                guard_seconds = max(0.0, guard_seconds - GUARD_DECAY_PER_SECOND * dt)

            is_prolonged_slouch = slouch_seconds >= PROLONGED_SLOUCH_SECONDS

            # ---- FUSION SCORING ----
            score = 1.0
            strain_weight = 1.0 if sustained_genuine_strain else 0.5
            score += facial_stress_smoothed * MAX_STRAIN_SCORE_CONTRIBUTION * strain_weight

            if is_guarding:
                score += GUARD_SCORE
            if is_slouching:
                score += SLOUCH_SCORE
            if is_prolonged_slouch:
                score += PROLONGED_SLOUCH_SCORE
            if is_limping:
                score += LIMP_SCORE
            if is_guarding and facial_stress_smoothed > COMBO_STRAIN_THRESHOLD:
                score += STRAIN_GUARD_COMBO_BONUS

            calculated_tension = int(np.clip(int(score + 0.5), 1, 10))
            tension_history.append(calculated_tension)
            final_tension_index = int(np.mean(tension_history) + 0.5) if tension_history else 1

            if final_tension_index >= 7:
                status_text, hud_color = "EMERGENCY CRITICAL DISTRESS", (0, 0, 255)
            elif final_tension_index >= 3:
                status_text, hud_color = "Monitoring: Discomfort / Fatigue", (0, 255, 255)
            else:
                status_text, hud_color = "Standard Seating / Normal", (0, 255, 0)

            cv2.putText(frame, f"Tension Rating: {final_tension_index}/10", (30, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, hud_color, 2)
            cv2.putText(frame, status_text, (30, 90), cv2.FONT_HERSHEY_DUPLEX, 0.6, hud_color, 1)
            cv2.putText(frame, f"Limp:{is_limping} Guard:{is_guarding}({guard_seconds:.2f}s) "
                                f"Slouch:{is_slouching}({slouch_seconds:.1f}s) Smile:{is_smiling}",
                        (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Strain:{facial_stress:.2f} Smoothed:{facial_stress_smoothed:.2f} "
                                f"Sustained:{sustained_genuine_strain}({strain_seconds:.2f}s)",
                        (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"FPS:{fps_smoothed:.0f} HeadPoseReliable:{head_pose_reliable}",
                        (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            cv2.imshow('ED Triage Assistant - Intelligent Fusion Monitor', frame)

        except Exception as e:
            # A single bad/corrupt frame or transient landmark glitch shouldn't kill the whole
            # session. Log it and keep going instead of crashing out mid-test.
            print(f"[Vision] Skipped a frame due to error: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    pose.close()
    print("Camera system closed safely.")


if __name__ == "__main__":
    run_vision_sensor()

Fusion Layer code : 
import base64
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np
import sounddevice as sd
import mediapipe as mp
from ws_bridge import Broadcaster

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# ============================================================================
# SHARED STATE (thread-safe handoff between audio thread and main/vision thread)
# ============================================================================

class SharedAudioState:
    """Thread-safe container for the latest audio sensor reading. The audio thread writes
    to this continuously; the vision loop reads a snapshot of it once per frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self.status = "Normal Ambient Sound"
        self.vol_peak = 0.0
        self.freq = 0.0
        self.last_update = time.time()
        self.alive = True  # flips to False if the audio thread dies, so the UI can show it

    def update(self, status, vol_peak, freq):
        with self._lock:
            self.status = status
            self.vol_peak = vol_peak
            self.freq = freq
            self.last_update = time.time()

    def mark_dead(self, reason):
        with self._lock:
            self.alive = False
            self.status = f"Audio Sensor Offline ({reason})"
            self.last_update = time.time()

    def snapshot(self):
        with self._lock:
            return self.status, self.vol_peak, self.freq, self.last_update, self.alive


audio_state = SharedAudioState()
stop_event = threading.Event()

# ============================================================================
# AUDIO THREAD (adapted from audio_sensor.py)
# ============================================================================

AUDIO_SAMPLE_RATE = 44100
AUDIO_CHUNK = 1024
AUDIO_CONSECUTIVE_FRAMES_REQUIRED = 3
AUDIO_NOISE_FLOOR_WINDOW = 100
AUDIO_ONSET_RATIO_REQUIRED = 3.0
AUDIO_FLATNESS_MIN_FOR_COUGH = 0.15


def _audio_features(audio_chunk):
    samples = (audio_chunk[:, 0] if audio_chunk.ndim > 1 else audio_chunk).astype(np.float64)

    rms = np.sqrt(np.mean(samples ** 2))
    vol_peak = rms * 1000

    windowed = samples * np.hanning(len(samples))
    fft_vals = np.fft.rfft(windowed)
    fft_freqs = np.fft.rfftfreq(len(windowed), d=1.0 / AUDIO_SAMPLE_RATE)
    magnitude = np.abs(fft_vals) + 1e-12

    freq = 0.0 if magnitude.max() < 1e-6 else fft_freqs[np.argmax(magnitude)]

    log_mag = np.log(magnitude)
    geo_mean = np.exp(np.mean(log_mag))
    arith_mean = np.mean(magnitude)
    flatness = geo_mean / arith_mean

    return vol_peak, freq, flatness


def audio_worker():
    """Runs continuously in a background thread, updating audio_state after every frame.
    If the input stream dies for any reason, audio_state is marked dead so the dashboard
    can show it instead of silently going stale."""
    try:
        noise_floor_history = deque(maxlen=AUDIO_NOISE_FLOOR_WINDOW)
        consecutive_hits = 0
        recent_freqs = deque(maxlen=AUDIO_CONSECUTIVE_FRAMES_REQUIRED)

        with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, blocksize=AUDIO_CHUNK) as stream:
            print("[Audio] Calibrating ambient noise for 3s... stay quiet.")
            for _ in range(int(3 * AUDIO_SAMPLE_RATE / AUDIO_CHUNK)):
                if stop_event.is_set():
                    return
                chunk, _ = stream.read(AUDIO_CHUNK)
                vol_peak, _, _ = _audio_features(chunk)
                noise_floor_history.append(vol_peak)
            print(f"[Audio] Calibration done. Initial noise floor: {np.mean(noise_floor_history):.1f}")

            while not stop_event.is_set():
                chunk, _ = stream.read(AUDIO_CHUNK)
                vol_peak, freq, flatness = _audio_features(chunk)

                noise_floor = np.median(noise_floor_history)
                onset_ratio = vol_peak / max(noise_floor, 1e-6)

                is_sudden_and_loud = onset_ratio > AUDIO_ONSET_RATIO_REQUIRED
                is_broadband = flatness > AUDIO_FLATNESS_MIN_FOR_COUGH

                recent_freqs.append(freq)

                if is_sudden_and_loud and is_broadband:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0
                    noise_floor_history.append(vol_peak)

                status = "Normal Ambient Sound"
                if consecutive_hits >= AUDIO_CONSECUTIVE_FRAMES_REQUIRED:
                    avg_freq = np.mean(recent_freqs)
                    if 600.0 <= avg_freq <= 1000.0:
                        status = "Wheezing / Stridor Detected"
                    elif 100.0 <= avg_freq <= 800.0:
                        status = "Severe Coughing Fit Detected"
                    else:
                        status = "Sudden Loud Noise (Unclassified)"
                elif onset_ratio > AUDIO_ONSET_RATIO_REQUIRED and not is_broadband:
                    status = "Steady Background Noise / Continuous Audio"

                audio_state.update(status, vol_peak, freq)

    except Exception as e:
        print(f"[Audio] Thread stopped due to error: {e}")
        audio_state.mark_dead(str(e))

# ============================================================================
# VISION (adapted from the refined vision_sensor.py) -- runs on main thread
# ============================================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_BROW, LEFT_EYE_TOP = 105, 159
RIGHT_BROW, RIGHT_EYE_TOP = 334, 386
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
FACE_TOP, FACE_BOTTOM = 10, 152

LM = mp_pose.PoseLandmark
CALIBRATION_SECONDS = 2.0
GAIT_WINDOW = 30
TENSION_SMOOTHING = 15

PROCESSING_WIDTH = 480

THRESHOLD_STD_MULTIPLIER = 3.0
MIN_SIGNAL_THRESHOLD = 0.03
MIN_SIGNALS_FOR_GENUINE_STRAIN = 2
STRONG_SIGNAL_MULTIPLIER = 1.8
SATURATION_MULTIPLIER = 1.8

SMILE_CORNER_LIFT_THRESHOLD = 0.015
SMILE_WIDTH_INCREASE_THRESHOLD = 0.08

HEAD_POSE_TOLERANCE = 0.30

SLOUCH_RATIO_THRESHOLD = 0.75
GUARD_PROXIMITY_RATIO = 0.5
LANDMARK_VISIBILITY_THRESHOLD = 0.4

WRIST_CLASP_MAX_DIST_RATIO = 0.35
SINGLE_WRIST_HORIZONTAL_RATIO = 0.65
SINGLE_WRIST_VERTICAL_RATIO = 2.0

GUARD_SECONDS_REQUIRED = 0.5
GUARD_DECAY_PER_SECOND = 1.5
STRAIN_BUILD_SECONDS_REQUIRED = 0.25
STRAIN_DECAY_PER_SECOND = 2.5
FACIAL_STRESS_RELEASE_PER_SECOND = 0.6
PROLONGED_SLOUCH_SECONDS = 3.0
SLOUCH_DECAY_PER_SECOND = 4.0

MAX_STRAIN_SCORE_CONTRIBUTION = 6.0
GUARD_SCORE = 3.0
SLOUCH_SCORE = 1.5
PROLONGED_SLOUCH_SCORE = 2.0
LIMP_SCORE = 3.0
STRAIN_GUARD_COMBO_BONUS = 2.0
COMBO_STRAIN_THRESHOLD = 0.3

AUDIO_WHEEZE_SCORE_BONUS = 3.0
AUDIO_COUGH_SCORE_BONUS = 2.0
AUDIO_STALE_SECONDS = 2.0


def _dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def eye_aspect_ratio(lm, eye_idx, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = pts
    vertical = (_dist(p2, p6) + _dist(p3, p5)) / 2.0
    horizontal = _dist(p1, p4)
    return vertical / horizontal if horizontal > 1e-6 else 0.0


def facial_raw_metrics(face_landmarks, w, h):
    lm = face_landmarks.landmark
    ear = (eye_aspect_ratio(lm, LEFT_EYE, w, h) + eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2

    face_height = max(abs(lm[FACE_BOTTOM].y * h - lm[FACE_TOP].y * h), 1e-6)
    brow_dist = (abs(lm[LEFT_BROW].y * h - lm[LEFT_EYE_TOP].y * h) +
                 abs(lm[RIGHT_BROW].y * h - lm[RIGHT_EYE_TOP].y * h)) / 2 / face_height

    mouth_h = _dist((lm[MOUTH_TOP].x * w, lm[MOUTH_TOP].y * h),
                    (lm[MOUTH_BOTTOM].x * w, lm[MOUTH_BOTTOM].y * h))
    mouth_openness = mouth_h / face_height

    mouth_width = _dist((lm[MOUTH_LEFT].x * w, lm[MOUTH_LEFT].y * h),
                        (lm[MOUTH_RIGHT].x * w, lm[MOUTH_RIGHT].y * h)) / face_height

    mouth_center_y = (lm[MOUTH_TOP].y * h + lm[MOUTH_BOTTOM].y * h) / 2
    corner_lift = ((mouth_center_y - lm[MOUTH_LEFT].y * h) +
                   (mouth_center_y - lm[MOUTH_RIGHT].y * h)) / 2 / face_height

    left_eye_center = np.mean([(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE], axis=0)
    right_eye_center = np.mean([(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE], axis=0)
    eye_distance = max(_dist(left_eye_center, right_eye_center), 1e-6)
    face_shape_ratio = face_height / eye_distance

    return ear, brow_dist, mouth_openness, mouth_width, corner_lift, face_shape_ratio


def compute_facial_stress_components(ear, brow_dist, mouth_openness, baseline):
    base_ear, base_brow, base_mouth_openness = baseline
    squint = max(0.0, (base_ear - ear) / max(base_ear, 1e-6))
    furrow = max(0.0, (base_brow - brow_dist) / max(base_brow, 1e-6))
    mouth_dev = abs(mouth_openness - base_mouth_openness)
    return squint, furrow, mouth_dev


def detect_smile(mouth_width, corner_lift, baseline_width, baseline_corner_lift):
    width_increase = mouth_width - baseline_width
    lift_increase = corner_lift - baseline_corner_lift
    return width_increase > SMILE_WIDTH_INCREASE_THRESHOLD and lift_increase > SMILE_CORNER_LIFT_THRESHOLD


def _saturate(raw_value, threshold):
    if threshold <= 1e-6:
        return 0.0
    cap = threshold * SATURATION_MULTIPLIER
    return float(np.clip(raw_value / cap, 0.0, 1.0))


def audio_score_contribution(status, last_update, alive):
    if not alive:
        return 0.0, status  # dead thread never contributes to the score, and the UI sees why
    if time.time() - last_update > AUDIO_STALE_SECONDS:
        return 0.0, "STALE"
    if status == "Wheezing / Stridor Detected":
        return AUDIO_WHEEZE_SCORE_BONUS, status
    if status == "Severe Coughing Fit Detected":
        return AUDIO_COUGH_SCORE_BONUS, status
    return 0.0, status


def run_fusion_layer():
    print("Starting Fusion Layer: Audio + Vision Triage Engine...")

    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()

    broadcaster = Broadcaster()
    broadcaster.start()

    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        stop_event.set()
        return

    tension_history = deque(maxlen=TENSION_SMOOTHING)
    left_ankle_y_hist = deque(maxlen=GAIT_WINDOW)
    right_ankle_y_hist = deque(maxlen=GAIT_WINDOW)

    guard_seconds = 0.0
    strain_seconds = 0.0
    slouch_seconds = 0.0
    facial_stress_smoothed = 0.0

    baseline = None
    baseline_torso = None
    baseline_wrist_dist = None
    baseline_mouth_width = None
    baseline_corner_lift = None
    baseline_face_shape_ratio = None
    squint_thresh = furrow_thresh = mouth_thresh = None
    calib_ear, calib_brow, calib_mouth = [], [], []
    calib_mouth_width, calib_corner_lift, calib_face_shape = [], [], []
    calib_torso, calib_wrist_dist = [], []

    calibration_start = None
    calibration_done_face = False
    calibration_done_pose = False

    print("Calibrating neutral face and posture for ~2s.")
    print("IMPORTANT: hold the SAME head angle/distance you'll use during testing.")
    print("Sit in your normal resting position (hands NOT near your stomach), relax your face, don't smile...")

    last_time = time.time()
    fps_smoothed = 30.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = max(0.0, min(now - last_time, 0.2))
            last_time = now
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * (1.0 / dt if dt > 1e-3 else fps_smoothed)

            try:
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                proc_scale = PROCESSING_WIDTH / w if w > PROCESSING_WIDTH else 1.0
                if proc_scale < 1.0:
                    proc_frame = cv2.resize(frame, (int(w * proc_scale), int(h * proc_scale)))
                else:
                    proc_frame = frame
                proc_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

                face_results = face_mesh.process(proc_rgb)
                pose_results = pose.process(proc_rgb)

                if calibration_start is None:
                    calibration_start = now
                calibrating = (now - calibration_start) < CALIBRATION_SECONDS

                facial_stress = 0.0
                is_limping = False
                is_guarding = False
                is_slouching = False
                is_smiling = False
                sustained_genuine_strain = False
                head_pose_reliable = True
                torso_span = None
                squint = furrow = mouth_dev = 0.0

                # ---- FACE ----
                if face_results.multi_face_landmarks:
                    face_lms = face_results.multi_face_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, face_lms, mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=1))

                    ear, brow_dist, mouth_openness, mouth_width, corner_lift, face_shape_ratio = \
                        facial_raw_metrics(face_lms, w, h)

                    if calibrating:
                        calib_ear.append(ear)
                        calib_brow.append(brow_dist)
                        calib_mouth.append(mouth_openness)
                        calib_mouth_width.append(mouth_width)
                        calib_corner_lift.append(corner_lift)
                        calib_face_shape.append(face_shape_ratio)
                        cv2.putText(frame, "Calibrating... HOLD STILL, DON'T SMILE", (30, 50),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        if not calibration_done_face and len(calib_ear) > 0:
                            baseline = (np.mean(calib_ear), np.mean(calib_brow), np.mean(calib_mouth))
                            baseline_mouth_width = np.mean(calib_mouth_width)
                            baseline_corner_lift = np.mean(calib_corner_lift)
                            baseline_face_shape_ratio = np.mean(calib_face_shape)

                            ear_std = np.std(calib_ear) / max(baseline[0], 1e-6)
                            brow_std = np.std(calib_brow) / max(baseline[1], 1e-6)
                            mouth_std = np.std(calib_mouth)

                            squint_thresh = max(ear_std * THRESHOLD_STD_MULTIPLIER, MIN_SIGNAL_THRESHOLD)
                            furrow_thresh = max(brow_std * THRESHOLD_STD_MULTIPLIER, MIN_SIGNAL_THRESHOLD)
                            mouth_thresh = max(mouth_std * THRESHOLD_STD_MULTIPLIER, MIN_SIGNAL_THRESHOLD)
                            calibration_done_face = True
                            print(f"[Vision] Face calibration done. Thresholds -> squint:{squint_thresh:.3f} "
                                  f"furrow:{furrow_thresh:.3f} mouth:{mouth_thresh:.3f}")

                        if baseline is not None:
                            shape_deviation = abs(face_shape_ratio - baseline_face_shape_ratio) / max(baseline_face_shape_ratio, 1e-6)
                            head_pose_reliable = shape_deviation <= HEAD_POSE_TOLERANCE

                            if head_pose_reliable:
                                is_smiling = detect_smile(mouth_width, corner_lift, baseline_mouth_width, baseline_corner_lift)
                                squint, furrow, mouth_dev = compute_facial_stress_components(ear, brow_dist, mouth_openness, baseline)

                                signals_active = sum([
                                    squint > squint_thresh,
                                    furrow > furrow_thresh,
                                    mouth_dev > mouth_thresh,
                                ])
                                strong_single_signal = (
                                    squint > squint_thresh * STRONG_SIGNAL_MULTIPLIER or
                                    furrow > furrow_thresh * STRONG_SIGNAL_MULTIPLIER or
                                    mouth_dev > mouth_thresh * STRONG_SIGNAL_MULTIPLIER
                                )
                                frame_has_genuine_strain = (
                                    (signals_active >= MIN_SIGNALS_FOR_GENUINE_STRAIN) or strong_single_signal
                                ) and not is_smiling

                                if frame_has_genuine_strain:
                                    strain_seconds += dt
                                else:
                                    strain_seconds = max(0.0, strain_seconds - STRAIN_DECAY_PER_SECOND * dt)
                                sustained_genuine_strain = strain_seconds >= STRAIN_BUILD_SECONDS_REQUIRED

                                if is_smiling:
                                    facial_stress = 0.0
                                else:
                                    norm_squint = _saturate(squint, squint_thresh)
                                    norm_furrow = _saturate(furrow, furrow_thresh)
                                    norm_mouth = _saturate(mouth_dev, mouth_thresh)
                                    raw_blend = 0.4 * norm_squint + 0.35 * norm_furrow + 0.25 * norm_mouth
                                    facial_stress = float(np.clip(raw_blend, 0.0, 1.0))

                if facial_stress > facial_stress_smoothed:
                    facial_stress_smoothed = facial_stress
                else:
                    facial_stress_smoothed = max(
                        facial_stress, facial_stress_smoothed - FACIAL_STRESS_RELEASE_PER_SECOND * dt
                    )

                # ---- POSE ----
                if pose_results.pose_landmarks:
                    lm = pose_results.pose_landmarks.landmark
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2))

                    l_sh, r_sh = lm[LM.LEFT_SHOULDER.value], lm[LM.RIGHT_SHOULDER.value]
                    l_hip, r_hip = lm[LM.LEFT_HIP.value], lm[LM.RIGHT_HIP.value]
                    l_wrist, r_wrist = lm[LM.LEFT_WRIST.value], lm[LM.RIGHT_WRIST.value]
                    l_ankle, r_ankle = lm[LM.LEFT_ANKLE.value], lm[LM.RIGHT_ANKLE.value]

                    hips_visible = (l_hip.visibility > LANDMARK_VISIBILITY_THRESHOLD and
                                    r_hip.visibility > LANDMARK_VISIBILITY_THRESHOLD)
                    wrists_visible = (l_wrist.visibility > LANDMARK_VISIBILITY_THRESHOLD or
                                      r_wrist.visibility > LANDMARK_VISIBILITY_THRESHOLD)
                    both_wrists_visible = (l_wrist.visibility > LANDMARK_VISIBILITY_THRESHOLD and
                                           r_wrist.visibility > LANDMARK_VISIBILITY_THRESHOLD)
                    ankles_visible = (l_ankle.visibility > LANDMARK_VISIBILITY_THRESHOLD and
                                      r_ankle.visibility > LANDMARK_VISIBILITY_THRESHOLD)

                    shoulder_mid_y = (l_sh.y + r_sh.y) / 2
                    shoulder_width = max(_dist((l_sh.x, l_sh.y), (r_sh.x, r_sh.y)), 1e-6)
                    hip_mid_y = (l_hip.y + r_hip.y) / 2
                    torso_span = max(abs(hip_mid_y - shoulder_mid_y), 1e-6)

                    if calibrating:
                        calib_torso.append(torso_span)
                    elif not calibration_done_pose and len(calib_torso) > 0:
                        baseline_torso = np.mean(calib_torso)
                        calibration_done_pose = True
                        print(f"[Vision] Posture calibration done. Baseline torso span={baseline_torso:.3f}")

                    if baseline_torso is not None:
                        is_slouching = torso_span < baseline_torso * SLOUCH_RATIO_THRESHOLD

                    if is_slouching:
                        slouch_seconds += dt
                    else:
                        slouch_seconds = max(0.0, slouch_seconds - SLOUCH_DECAY_PER_SECOND * dt)

                    hip_based_guard = False
                    if hips_visible and wrists_visible:
                        abdomen = ((l_hip.x + r_hip.x) / 2, hip_mid_y - torso_span * 0.15)
                        current_min_wrist_dist = min(_dist((l_wrist.x, l_wrist.y), abdomen),
                                                     _dist((r_wrist.x, r_wrist.y), abdomen))
                        if calibrating:
                            calib_wrist_dist.append(current_min_wrist_dist)
                        elif baseline_wrist_dist is None and len(calib_wrist_dist) > 0:
                            baseline_wrist_dist = np.mean(calib_wrist_dist)
                            print(f"[Vision] Guard calibration done. Baseline dist={baseline_wrist_dist:.3f}")
                        if baseline_wrist_dist is not None:
                            hip_based_guard = current_min_wrist_dist < baseline_wrist_dist * GUARD_PROXIMITY_RATIO

                    clasp_guard = False
                    if both_wrists_visible:
                        wrist_gap = _dist((l_wrist.x, l_wrist.y), (r_wrist.x, r_wrist.y))
                        torso_center_x = (l_sh.x + r_sh.x) / 2
                        wrists_center_x = (l_wrist.x + r_wrist.x) / 2
                        wrists_near_center = abs(wrists_center_x - torso_center_x) < shoulder_width * 0.6
                        clasp_guard = (wrist_gap < shoulder_width * WRIST_CLASP_MAX_DIST_RATIO) and wrists_near_center

                    single_wrist_hug = False
                    torso_center_x = (l_sh.x + r_sh.x) / 2
                    for wrist, vis in ((l_wrist, l_wrist.visibility), (r_wrist, r_wrist.visibility)):
                        if vis <= LANDMARK_VISIBILITY_THRESHOLD:
                            continue
                        horiz_ok = abs(wrist.x - torso_center_x) < shoulder_width * SINGLE_WRIST_HORIZONTAL_RATIO
                        vert_ok = (wrist.y > shoulder_mid_y and
                                   wrist.y < shoulder_mid_y + shoulder_width * SINGLE_WRIST_VERTICAL_RATIO)
                        if horiz_ok and vert_ok:
                            single_wrist_hug = True
                            break

                    wrist_near = hip_based_guard or clasp_guard or single_wrist_hug

                    if wrist_near:
                        guard_seconds += dt
                    else:
                        guard_seconds = max(0.0, guard_seconds - GUARD_DECAY_PER_SECOND * dt)
                    is_guarding = guard_seconds >= GUARD_SECONDS_REQUIRED

                    if ankles_visible:
                        left_ankle_y_hist.append(l_ankle.y)
                        right_ankle_y_hist.append(r_ankle.y)
                        if len(left_ankle_y_hist) == GAIT_WINDOW:
                            l_range = max(left_ankle_y_hist) - min(left_ankle_y_hist)
                            r_range = max(right_ankle_y_hist) - min(right_ankle_y_hist)
                            total_motion = l_range + r_range
                            if total_motion > 0.05:
                                asymmetry = abs(l_range - r_range) / max(total_motion, 1e-6)
                                is_limping = asymmetry > 0.35
                else:
                    slouch_seconds = max(0.0, slouch_seconds - SLOUCH_DECAY_PER_SECOND * dt)
                    guard_seconds = max(0.0, guard_seconds - GUARD_DECAY_PER_SECOND * dt)

                is_prolonged_slouch = slouch_seconds >= PROLONGED_SLOUCH_SECONDS

                # ---- FETCH LATEST AUDIO SNAPSHOT ----
                audio_status, audio_vol, audio_freq, audio_last_update, audio_alive = audio_state.snapshot()
                audio_bonus, audio_display_status = audio_score_contribution(audio_status, audio_last_update, audio_alive)

                # ---- FUSION SCORING (vision + audio combined) ----
                score = 1.0
                strain_weight = 1.0 if sustained_genuine_strain else 0.5
                score += facial_stress_smoothed * MAX_STRAIN_SCORE_CONTRIBUTION * strain_weight

                if is_guarding:
                    score += GUARD_SCORE
                if is_slouching:
                    score += SLOUCH_SCORE
                if is_prolonged_slouch:
                    score += PROLONGED_SLOUCH_SCORE
                if is_limping:
                    score += LIMP_SCORE
                if is_guarding and facial_stress_smoothed > COMBO_STRAIN_THRESHOLD:
                    score += STRAIN_GUARD_COMBO_BONUS

                score += audio_bonus

                calculated_tension = int(np.clip(int(score + 0.5), 1, 10))
                tension_history.append(calculated_tension)
                final_tension_index = int(np.mean(tension_history) + 0.5) if tension_history else 1

                if final_tension_index >= 7:
                    status_text, hud_color = "EMERGENCY CRITICAL DISTRESS", (0, 0, 255)
                elif final_tension_index >= 3:
                    status_text, hud_color = "Monitoring: Discomfort / Fatigue", (0, 255, 255)
                else:
                    status_text, hud_color = "Standard Seating / Normal", (0, 255, 0)

                cv2.putText(frame, f"FUSED Tension Rating: {final_tension_index}/10", (30, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, hud_color, 2)
                cv2.putText(frame, status_text, (30, 90), cv2.FONT_HERSHEY_DUPLEX, 0.6, hud_color, 1)
                cv2.putText(frame, f"Limp:{is_limping} Guard:{is_guarding}({guard_seconds:.2f}s) "
                                    f"Slouch:{is_slouching}({slouch_seconds:.1f}s) Smile:{is_smiling}",
                            (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Strain:{facial_stress:.2f} Smoothed:{facial_stress_smoothed:.2f} "
                                    f"Sustained:{sustained_genuine_strain}({strain_seconds:.2f}s)",
                            (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                audio_color = (0, 0, 255) if not audio_alive else (0, 200, 255)
                cv2.putText(frame, f"AUDIO: {audio_display_status} (Vol:{audio_vol:.1f} Freq:{audio_freq:.0f}Hz "
                                    f"Bonus:+{audio_bonus:.1f})",
                            (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, audio_color, 1)
                cv2.putText(frame, f"FPS:{fps_smoothed:.0f} HeadPoseReliable:{head_pose_reliable}",
                            (30, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

                cv2.imshow('ED Triage Assistant - Fusion Layer (Audio + Vision)', frame)

                # ---- Push live data to the dashboard ----
                color = 'green' if final_tension_index < 3 else ('amber' if final_tension_index < 7 else 'red')
                code = color.upper()
                frame_b64 = None
                if int(now * 10) % 3 == 0:  # throttle frame streaming to ~1/3 of processed frames
                    small = cv2.resize(frame, (480, int(480 * h / w)))
                    ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    if ok:
                        frame_b64 = base64.b64encode(buf).decode('utf-8')
                broadcaster.send({
                    "score": int(final_tension_index),
                    "label": status_text,
                    "color": color,
                    "code": code,
                    "guard": {"active": bool(is_guarding), "hold": round(float(guard_seconds), 2)},
                    "slouch": {"active": bool(is_slouching), "duration": round(float(slouch_seconds), 1)},
                    "limp": {"active": bool(is_limping)},
                    "strain": {"value": round(float(facial_stress_smoothed), 2), "sustained": bool(sustained_genuine_strain)},
                    "audio": {
                        "status": audio_display_status,
                        "vol": float(audio_vol),
                        "freq": float(audio_freq),
                        "alive": bool(audio_alive),
                    },
                    "frame_b64": frame_b64,
                })

            except Exception as e:
                print(f"[Vision] Skipped a frame due to error: {e}")
                traceback.print_exc()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        pose.close()
        audio_thread.join(timeout=2.0)
        print("Fusion layer shut down cleanly.")


if __name__ == "__main__":
    run_fusion_layer()

Ws bridge code :
import asyncio
import json
import threading
import numpy as np
import websockets


class _NumpySafeEncoder(json.JSONEncoder):
    """Fallback safety net: converts numpy scalar types (bool_, int64, float32, etc.)
    to native Python types if any slip into a payload without being cast by hand."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class Broadcaster:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self._clients = set()
        self._loop = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        self._ready.wait()  # block until the server is actually listening

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        start_server = websockets.serve(self._handler, self.host, self.port)
        self._loop.run_until_complete(start_server)
        print(f"[Bridge] WebSocket server running at ws://{self.host}:{self.port}")
        self._ready.set()
        self._loop.run_forever()

    async def _handler(self, websocket, path=None):
        self._clients.add(websocket)
        print(f"[Bridge] Dashboard connected ({len(self._clients)} total)")
        try:
            async for _ in websocket:
                pass  # broadcast-only channel; incoming messages are ignored
        finally:
            self._clients.discard(websocket)
            print(f"[Bridge] Dashboard disconnected ({len(self._clients)} total)")

    def send(self, data: dict):
        """Thread-safe: call this directly from your synchronous main loop."""
        if self._loop is None:
            return
        payload = json.dumps(data, cls=_NumpySafeEncoder)
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    async def _broadcast(self, payload):
        if not self._clients:
            return
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


