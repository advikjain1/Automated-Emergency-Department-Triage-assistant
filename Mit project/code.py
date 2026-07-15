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
import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
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

# --- All hold/decay behavior is TIME-based (seconds), not frame-count-based. Seconds behave
# the same regardless of FPS. ---
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

# --- Multi-person handling ---
# Neither FaceMesh(max_num_faces=1) nor Pose() track *identity* across frames. If a second
# person enters the frame, either model can silently start reporting a different person's
# face/body than the one we calibrated on. Because the score is additive across independent
# behavior flags (facial strain + guard + slouch + limp), a single frame that mixes signals
# from two different people can trip several flags at once and spike the score instantly.
# We defend against this in two ways:
#   1. A cheap, separate face-count check: if more than one face is visible for a short
#      sustained period, we FREEZE all scoring (not decay it -- freezing is safer than
#      decaying, since decaying could quietly hide a real ongoing emergency just because a
#      second person walked past the camera).
#   2. A per-frame sanity check on the pose skeleton's torso position: if it jumps further
#      than a person could plausibly move in that time, we treat that single frame as a
#      likely identity swap / occlusion glitch and skip using its pose data -- this catches
#      cases even when only one face happens to be visible (e.g. someone turned away).
MULTI_PERSON_FACE_COUNT_THRESHOLD = 1        # more than this many faces = multiple people
MULTI_PERSON_SECONDS_REQUIRED = 0.3          # must see >1 face for this long before freezing
MULTI_PERSON_CLEAR_SECONDS_REQUIRED = 0.6    # must see <=1 face for this long before resuming
POSE_MAX_CENTROID_JUMP_PER_SECOND = 0.6      # normalized-units/sec; above this = likely a body swap


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
    # Separate, lightweight detector used ONLY to count how many faces are visible. This is
    # much cheaper than running FaceMesh with a higher max_num_faces, which would cost more
    # compute per face and slow things down right when the room is busiest.
    face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
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

    # Multi-person guard state
    multi_person_present_seconds = 0.0
    multi_person_clear_seconds = 0.0
    is_multi_person_holding = False
    prev_torso_center = None

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

    # Defaults so the HUD/broadcast has something sane to show before the first real frame,
    # and so a frame that starts in a "holding" state doesn't crash on an undefined variable.
    final_tension_index = 1
    status_text, hud_color = "Standard Seating / Normal", (0, 255, 0)

    print("Calibrating neutral face and posture for ~2s.")
    print("IMPORTANT: hold the SAME head angle/distance you'll use during testing.")
    print("Sit in your normal resting position (hands NOT near your stomach), relax your face, don't smile...")
    print("If a second person is visible during calibration, calibration will simply pause until they leave.")

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

            # ---- MULTI-PERSON CHECK (runs every frame, cheap) ----
            face_count_results = face_detector.process(proc_rgb)
            num_faces_detected = len(face_count_results.detections) if face_count_results.detections else 0

            if num_faces_detected > MULTI_PERSON_FACE_COUNT_THRESHOLD:
                multi_person_present_seconds += dt
                multi_person_clear_seconds = 0.0
            else:
                multi_person_clear_seconds += dt
                multi_person_present_seconds = 0.0

            if not is_multi_person_holding and multi_person_present_seconds >= MULTI_PERSON_SECONDS_REQUIRED:
                is_multi_person_holding = True
                print(f"[Vision] {num_faces_detected} faces detected -- freezing score until frame clears.")
            elif is_multi_person_holding and multi_person_clear_seconds >= MULTI_PERSON_CLEAR_SECONDS_REQUIRED:
                is_multi_person_holding = False
                print("[Vision] Frame clear of extra people -- resuming normal scoring.")

            if calibration_start is None:
                calibration_start = now
            calibrating = (now - calibration_start) < CALIBRATION_SECONDS

            if is_multi_person_holding:
                # FROZEN FRAME: we still run FaceMesh/Pose purely so the operator can see the
                # skeleton overlay on screen, but nothing from this frame is allowed to touch
                # calibration or the running score. facial_stress_smoothed, guard_seconds,
                # slouch_seconds, strain_seconds, tension_history, and final_tension_index all
                # simply carry over unchanged from the last reliable frame.
                face_results = face_mesh.process(proc_rgb)
                pose_results = pose.process(proc_rgb)
                if face_results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, face_results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 165, 255), thickness=1))
                if pose_results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 165, 255), thickness=1, circle_radius=2))

                cv2.putText(frame, f"MULTIPLE PEOPLE DETECTED ({num_faces_detected}) -- SCORE HELD", (30, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 165, 255), 2)
                cv2.putText(frame, f"Last Known Tension Rating: {final_tension_index}/10", (30, 90),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, hud_color, 1)
                cv2.putText(frame, f"FPS:{fps_smoothed:.0f}", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

                cv2.imshow('ED Triage Assistant - Intelligent Fusion Monitor', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue  # skip everything below -- no calibration, no scoring, this frame is done

            # ---- Below this point: normal single-person path (unchanged logic) ----
            face_results = face_mesh.process(proc_rgb)
            pose_results = pose.process(proc_rgb)

            facial_stress = 0.0
            is_limping = False
            is_guarding = False
            is_slouching = False
            is_smiling = False
            sustained_genuine_strain = False
            head_pose_reliable = True
            pose_identity_reliable = True
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
                hip_mid_x = (l_hip.x + r_hip.x) / 2
                hip_mid_y = (l_hip.y + r_hip.y) / 2
                torso_span = max(abs(hip_mid_y - shoulder_mid_y), 1e-6)

                # --- Identity/occlusion glitch filter -------------------------------------
                # Even with only one face visible, Pose's single-person tracking can snap to a
                # different body (e.g. a second person partially in frame, or the patient
                # briefly occluded). Check how fast the torso centroid moved since last frame;
                # a jump faster than a seated person could plausibly move signals we may be
                # looking at a different body than a moment ago, so we skip using this one
                # frame's pose data rather than let it corrupt guard/slouch/limp detection.
                torso_center = (hip_mid_x, hip_mid_y)
                if prev_torso_center is not None and dt > 1e-3:
                    centroid_jump_speed = _dist(torso_center, prev_torso_center) / dt
                    pose_identity_reliable = centroid_jump_speed <= POSE_MAX_CENTROID_JUMP_PER_SECOND
                prev_torso_center = torso_center

                if pose_identity_reliable:
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
                        abdomen = (hip_mid_x, hip_mid_y - torso_span * 0.15)
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
                    # Glitch frame: keep is_guarding/is_slouching/is_limping at whatever they
                    # were last frame (declared False above only resets local display flags;
                    # the *_seconds accumulators, which drive the actual booleans, are simply
                    # not touched this frame). We recompute the booleans from the untouched
                    # accumulators so the HUD still reflects the true held state, not a reset.
                    is_slouching = baseline_torso is not None and slouch_seconds > 0 and \
                        (slouch_seconds >= SLOUCH_DECAY_PER_SECOND * 0) and torso_span is not None and \
                        torso_span < baseline_torso * SLOUCH_RATIO_THRESHOLD if baseline_torso else False
                    is_guarding = guard_seconds >= GUARD_SECONDS_REQUIRED
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
            cv2.putText(frame, f"FPS:{fps_smoothed:.0f} HeadPoseReliable:{head_pose_reliable} "
                                f"PoseIdentityReliable:{pose_identity_reliable}",
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
    face_detector.close()
    pose.close()
    print("Camera system closed safely.")


if __name__ == "__main__":
    run_vision_sensor()

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


