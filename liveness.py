import numpy as np
import face_recognition

def eye_aspect_ratio(eye):
    # compute the euclidean distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    # EAR formula
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blink(face_landmarks):
    # face_landmarks is a dict from face_recognition.face_landmarks
    # returns True if a blink (eyes closed) is detected based on EAR threshold
    left = np.array(face_landmarks.get('left_eye', []), dtype=np.float32)
    right = np.array(face_landmarks.get('right_eye', []), dtype=np.float32)
    
    if left.size == 0 or right.size == 0:
        return False, 0.0
    
    left_ear = eye_aspect_ratio(left)
    right_ear = eye_aspect_ratio(right)
    ear = (left_ear + right_ear) / 2.0
    
    # Typical EAR thresholds: closed ~ 0.15, open ~ 0.3
    return (ear < 0.20), ear

def liveness_detection(frame_rgb):
    # frame_rgb: image in RGB (face_recognition uses RGB)
    # returns (is_live, details)
    faces = face_recognition.face_locations(frame_rgb)
    
    if not faces:
        return False, {'reason': 'no_face_detected'}
    
    landmarks_list = face_recognition.face_landmarks(frame_rgb, faces)
    
    # check each face for blink (closed eyes)
    for landmarks in landmarks_list:
        blinked, ear = detect_blink(landmarks)
        if blinked:
            return True, {'ear': ear, 'method': 'blink_ear'}
    
    return False, {'reason': 'no_blink_detected'}

def check_liveness_quick(frame_rgb, timeout=5):
    """
    Quick liveness check on a single frame or short sequence.
    Returns True if blink detected.
    """
    is_live, info = liveness_detection(frame_rgb)
    return is_live, info

if __name__ == '__main__':
    print("Liveness detection module loaded.")