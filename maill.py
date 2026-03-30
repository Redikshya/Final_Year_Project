import os
from flask import Flask, render_template_string
from flask_mail import Mail, Message

# --- Configuration Notes ---
# 1. Install necessary libraries: pip install Flask Flask-Mail
# 2. You MUST replace the 'MAIL_PASSWORD' below with the 16-character
#    App Password you generated in your Google account settings.

app = Flask(__name__)

# --- CONFIGURATION FOR GMAIL/GOOGLE WORKSPACE ---
# Since your email is likely managed by Google, we use these settings:
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'youremail@gmail.com'  # Your sending email address (and also the recipient in this demo)

# VVVVVVVV  REPLACE THIS WITH YOUR 16-CHARACTER APP PASSWORD  VVVVVVVV
app.config['MAIL_PASSWORD'] = 'abcd efgh ijkl mnop' # <-- Placeholder for your App Password
# ^^^^^^^^  REPLACE THIS WITH YOUR 16-CHARACTER APP PASSWORD  ^^^^^^^^

app.config['MAIL_DEFAULT_SENDER'] = 'Flask Demo <noreply@yourdomain.com>'

mail = Mail(app)

@app.route('/send-email')
def send_email():
    """Handles the email sending logic."""
    recipient = 'youremail@gmail.com' # The email will be sent here

    msg = Message(
        subject='[Passcode] Hello Admin!',
        recipients=[recipient],
        html=f"""
        <div style="font-family: sans-serif; padding: 25px; background-color: #f3f4f6; border-radius: 10px;">
            <h2 style="color: #2563eb; border-bottom: 2px solid #3b82f6; padding-bottom: 10px;">Hello Redikshya!</h2>
            <p style="font-size: 16px; line-height: 1.5;">This email was successfully sent from your Flask application, *ap.py*, using your confirmed App Password and Flask-Mail.</p>
            <p style="font-size: 16px; line-height: 1.5;">You are now ready to integrate email functionality into your Lockify project!</p>
            <p style="margin-top: 20px;">Regards,<br>Your Flask Server</p>
        </div>
        """,
        body="Hello Admin! Your passcode is blahblah"
    )

    try:
        # Send the message
        mail.send(msg)
        print(f"Successfully sent email to {recipient}")
        return f'<div style="font-family: sans-serif; text-align: center; padding: 50px;"><h1 style="color: #10b981;">Success!</h1><p>Email sent to {recipient}. Check your inbox.</p><p><a href="/" style="color: #2563eb;">Go Back</a></p></div>'
    except Exception as e:
        # Log the error and return a failure message
        print(f"ERROR: Email failed to send. Details: {e}")
        # Display a custom error message on the screen
        return f'<div style="font-family: sans-serif; text-align: center; padding: 50px;"><h1 style="color: #ef4444;">Error!</h1><p>Email failed to send. This usually means the App Password or SMTP settings are wrong.</p><p>Error Detail: {e}</p><p><a href="/" style="color: #2563eb;">Go Back</a></p></div>'


if __name__ == '__main__':
    app.run(debug=True)








#liveness
# liveness.py
# Simple liveness detection using Eye Aspect Ratio (EAR) to detect blinks.
# This implementation uses face_recognition (dlib landmarks) and OpenCV only.

import time
import cv2
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

def detect_blink_from_face_landmarks(face_landmarks):
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

def liveness_detection_from_frame(frame_rgb):
    # frame_rgb: image in RGB (face_recognition uses RGB)
    # returns (is_live, details)
    faces = face_recognition.face_locations(frame_rgb)
    
    if not faces:
        return False, {'reason': 'no_face_detected'}
    
    landmarks_list = face_recognition.face_landmarks(frame_rgb, faces)
    
    # check each face for blink (closed eyes)
    for landmarks in landmarks_list:
        blinked, ear = detect_blink_from_face_landmarks(landmarks)
        if blinked:
            return True, {'ear': ear, 'method': 'blink_ear'}
    
    # if no blink detected, still consider further heuristics could be added
    return False, {'reason': 'no_blink_detected'}

def check_liveness_quick(frame_rgb, timeout=5):
    """
    Quick liveness check on a single frame or short sequence.
    Returns True if blink detected.
    """
    is_live, info = liveness_detection_from_frame(frame_rgb)
    return is_live, info

if __name__ == '__main__':
    print("Liveness detection module loaded.")





#app.py
import os
import shutil
import time
from datetime import datetime, date
import cv2
import numpy as np
from flask import Flask, render_template, redirect, request, session, url_for
from dotenv import load_dotenv
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from custom_knn import CustomKNeighborsClassifier
from model_evaluator import ModelEvaluator
import joblib
from flask_mysqldb import MySQL
import MySQLdb.cursors
import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask_mail import Mail, Message
import threading
import traceback
from liveness import liveness_detection_from_frame
import time


# Load environment variables
load_dotenv()

# Initialize Flask app
app=Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Flask error handler
@app.errorhandler(404)
@app.errorhandler(401)
@app.errorhandler(500)
def http_error_handler(error):
    return render_template('Error.html')

# MySQL configurations
app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "lockify_db"

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'youremail@gmail.com'

app.config['MAIL_PASSWORD'] = 'abcd efgh ijkl mnop'

app.config['MAIL_DEFAULT_SENDER'] = 'Flask Demo <noreply@yourdomain.com>'
mail = Mail(app)

mysql = MySQL(app)

with app.app_context():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT DATABASE();")
        print("Connected to:", cur.fetchone())
        cur.close()
    except Exception as e:
        print("Database connection failed:", e)

# Flask assign admin
"""@app.before_request
def before_request():
    g.user = None
    if 'admin' in session:
        g.user = session['admin']"""

# Current Date & Time
datetoday = datetime.today().strftime("%d-%m-%Y")
datetoday2 = datetime.today().strftime("%d %B %Y")

# Capture the video
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# ======= Check and Make Folders ========
folders = ['static/faces', 'final_model']
for folder in folders:
    os.makedirs(folder, exist_ok=True)
        
# ======= Total Registered Users ========
def totalreg():
    faces_dir = 'static/faces'
    if not os.path.isdir(faces_dir):
        return 0
    return len([name for name in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, name))])

# ======= Extract HOG Features from Face Image ========
def extract_hog_features(face_img):
    try:
        # Convert BGR to grayscale for HOG
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG features with optimized parameters for face recognition
        hog_features = hog(
            gray_face,
            orientations=8,           # Number of orientation bins
            pixels_per_cell=(8, 8),   # Size of cells for gradient computation
            cells_per_block=(2, 2),   # Size of blocks for normalization
            block_norm='L2-Hys',      # Block normalization method
            visualize=False,           # Don't return visualization
            feature_vector=True        # Return as 1D array
        )
        
        return hog_features
        
    except Exception as e:
        print(f"[ERROR] extract_hog_features: Failed to extract HOG features - {e}")
        return face_img.ravel()

# ======= Get Face From Image =========
def extract_faces(img):
    try:
        if img is None:
            print("[DEBUG] extract_faces: Input image is None")
            return np.array([])
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray_img, 1.5, 7)
        if len(face_points) > 0:
            print(f"[DEBUG] extract_faces: Detected {len(face_points)} face(s)")
        return face_points
    except Exception as e:
        print(f"[ERROR] extract_faces: Face detection failed - {e}")
        return np.array([])

# ======= Train Model Using Available Faces ========
def train_model():
    print("[INFO] train_model: Starting model training...")
    
    if 'face_recognition_model.pkl' in os.listdir('static'):
        print("[INFO] train_model: Removing old model file...")
        os.remove('static/face_recognition_model.pkl')

    if len(os.listdir('static/faces')) == 0:
        print("[WARNING] train_model: No face images found in static/faces/")
        return

    print(f"[INFO] train_model: Found {len(os.listdir('static/faces'))} user directories")
    
    faces = []
    labels = []
    user_list = os.listdir('static/faces')
    
    for user in user_list:
        user_images = os.listdir(f'static/faces/{user}')
        print(f"[INFO] train_model: Processing user '{user}' with {len(user_images)} images")
        
        for img_name in user_images:
            img_path = f'static/faces/{user}/{img_name}'
            img = cv2.imread(img_path)
            if img is not None:
                resized_face = cv2.resize(img, (50, 50))
                try:
                    hog_features = extract_hog_features(resized_face)
                    faces.append(hog_features)
                    labels.append(user)
                    print(f"[DEBUG] train_model: Successfully extracted HOG features for {img_path}")
                except Exception as e:
                    print(f"[ERROR] train_model: Failed to extract HOG features for {img_path}: {e}")
                    # Skip this image if HOG extraction fails
                    continue
            else:
                print(f"[WARNING] train_model: Failed to load image {img_path}")

    print(f"[INFO] train_model: Total faces processed: {len(faces)}")
    print(f"[INFO] train_model: Total labels: {len(labels)}")
    
    if len(faces) == 0:
        print("[ERROR] train_model: No valid faces found for training")
        return

    faces = np.array(faces)
    labels = np.array(labels)
    
    print(f"[INFO] train_model: Training KNN model with {len(faces)} samples...")
    print(f"[INFO] train_model: HOG feature vector length: {faces.shape[1]} features per face")
    print(f"[INFO] train_model: Total features: {faces.shape[0] * faces.shape[1]} values")
    
    # Split data into training and testing sets
    print(f"[INFO] train_model: Splitting data into train/test sets...")
    try:
        # Use 80% for training, 20% for testing
        # stratify ensures each user has proportional representation in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            faces, labels, 
            test_size=0.2,           # 20% for testing
            random_state=42,          # For reproducible results
            stratify=labels           # Maintain class balance
        )
        
    except Exception as e:
        print(f"[WARNING] train_model: Train/test split failed, using all data for training: {e}")
        X_train, X_test, y_train, y_test = faces, np.array([]), labels, np.array([])
    
    # Train the model on training data only
    print(f"[INFO] train_model: Training KNN model on {len(X_train)} samples...")
    knn = CustomKNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Evaluate model performance on both training and testing data
    print(f"[INFO] train_model: Evaluating model performance...")
    evaluator = ModelEvaluator()
    
    # Evaluate on training data (to check for overfitting)
    train_metrics = evaluator.evaluate_model(knn, X_train, y_train, "Custom_KNN_Train")
    
    # Evaluate on testing data (unseen data - more realistic performance)
    if len(X_test) > 0:
        test_metrics = evaluator.evaluate_model(knn, X_test, y_test, "Custom_KNN_Test")
        print(f"[INFO] train_model: Training accuracy: {train_metrics['accuracy']:.2%}")
        print(f"[INFO] train_model: Testing accuracy: {test_metrics['accuracy']:.2%}")
    else:
        print(f"[INFO] train_model: Training accuracy: {train_metrics['accuracy']:.2%}")
        print(f"[WARNING] train_model: No test data available for validation")
    
    # Clean up old results (keep only latest 3)
    evaluator.cleanup_old_results(keep_latest=3)
    
    print(f"[SUCCESS] train_model: Model evaluation completed")
    
    # Log HOG feature improvements
    old_feature_count = 50 * 50  # Old raw pixel count
    new_feature_count = faces.shape[1] if len(faces) > 0 else 0
    reduction = ((old_feature_count - new_feature_count) / old_feature_count) * 100
    print(f"[INFO] train_model: Feature reduction: {old_feature_count} → {new_feature_count} ({reduction:.1f}% reduction)")
    print(f"[INFO] train_model: HOG features provide more robust face representation")
    
    # Final model summary
    print(f"\n{'='*60}")
    print(f"MODEL TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(faces)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    
    print(f"{'='*60}")
    
    model_path = 'static/face_recognition_model.pkl'
    joblib.dump(knn, model_path)
    print(f"[SUCCESS] train_model: Model saved to {model_path}")
    print(f"[INFO] train_model: Model trained on {len(set(labels))} unique users")



# ======= Get Registered Users List ========
def get_registered_users():
    """Get list of all registered users from database."""
    try:
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("SELECT id FROM student WHERE status='registered'")
        registered_ids = {str(row['id']) for row in cur.fetchall()}
        cur.close()
        return registered_ids
    except Exception as e:
        print(f"[ERROR] get_registered_users: Failed to get registered users - {e}")
        return set()

# ======= Identify Face Using KNN Model ========
def identify_face(face_array):
    """
    Identify face using trained KNN model.
    If identified person is not in registered users list, mark as Unknown.
    """
    try:
        model_path = 'static/face_recognition_model.pkl'
        if not os.path.exists(model_path):
            print(f"[WARNING] identify_face: Model file not found at {model_path}")
            return ["Unknown"]
        
        print(f"[DEBUG] identify_face: Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Get prediction from model
        prediction = model.predict(face_array)
        predicted_person = prediction[0]
        
        # Check if predicted person is actually registered
        if '$' in predicted_person:
            # Extract user ID from prediction (format: Name$ID$)
            parts = predicted_person.split('$')
            if len(parts) >= 2:
                predicted_user_id = parts[1]
                
                # Get list of registered users
                registered_users = get_registered_users()
                
                # Check if predicted user is registered
                if predicted_user_id in registered_users:
                    print(f"[INFO] identify_face: Person identified and verified as registered user: {predicted_person}")
                    return prediction
                else:
                    
                    print(f"[INFO] identify_face: Marking as 'Unknown' for security")
                    return ["Unknown"]
            else:
                print(f"[WARNING] identify_face: Invalid prediction format: {predicted_person}")
                return ["Unknown"]
        else:
            # Prediction doesn't have expected format
            print(f"[WARNING] identify_face: Prediction format not recognized: {predicted_person}")
            return ["Unknown"]
        
    except Exception as e:
        print(f"[ERROR] identify_face: Face identification failed - {e}")
        return ["Unknown"]

# ======= Remove Access of Deleted User ======
def remAccess():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # Collect valid IDs from both user tables
    cur.execute("SELECT id FROM student WHERE status='registered'")
    registered_ids = {str(row['id']) for row in cur.fetchall()}


    valid_ids = registered_ids 

    # If there are valid IDs, remove all access records that don't belong to them
    if valid_ids:
        ids_str = ",".join([f"'{i}'" for i in valid_ids])
        cur.execute(f"DELETE FROM access WHERE id NOT IN ({ids_str})")

    mysql.connection.commit()
    cur.close()

# ======== Get Info From Access File =========
def extract_access():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    datetoday_mysql = date.today().strftime("%Y-%m-%d")
        
    query = """
        SELECT a.name, a.id, a.time,
               COALESCE(s.status, 'Unknown') AS status
        FROM access a
        LEFT JOIN student s ON a.id = s.id
        WHERE DATE(a.time) = %s
        ORDER BY a.time ASC
    """
    cur.execute(query, (datetoday_mysql,))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return [], [], [],  datetoday, [], 0

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    times = [r['time'].strftime("%H:%M:%S") for r in rows]
    reg   = [r['status'] for r in rows]
    l     = len(rows)

    return names, rolls,  times, datetoday, reg, l

# ======== Save Access =========
def add_access(name):
    try:
        print(f"[INFO] add_access: Processing access for '{name}'")
        
        if '$' not in name:
            print(f"[ERROR] add_access: Invalid name format: {name}")
            return
            
        parts = name.split('$')
        if len(parts) < 2:
            print(f"[ERROR] add_access: Insufficient name parts: {name}")
            return
            
        username = parts[0]
        userid = parts[1]
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"[INFO] add_access: User: {username}, ID: {userid}")
        print(f"[INFO] add_access: Current time: {current_datetime}")

        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        # Check if accessed today (ignoring time, only by DATE)
        print(f"[DEBUG] add_access: Checking if user {userid} has access today")
        cur.execute("""
            SELECT * FROM access 
            WHERE id=%s AND DATE(time)=%s
    """, (userid, datetime.now().strftime("%Y-%m-%d")))
        already = cur.fetchone()

        if already:
            print(f"[INFO] add_access: User {userid} already accessed today")
            cur.close()
            return 

        # Insert new access with full date+time
        print(f"[INFO] add_access: Inserting new access record for user {userid}")
        cur.execute("""
            INSERT INTO access (id, name, time)
            VALUES (%s, %s,  %s)
        """, (userid, username, current_datetime))
        mysql.connection.commit()
        cur.close()
        print(f"[SUCCESS] add_access: Access recorded successfully for user {userid}")
    except Exception as e:
        print(f"[ERROR] add_access: Failed to add access - {e}")


# ======= Flask Access Page =========
@app.route('/access')
def take_access():
    # Fetch today's access from MySQL
    names, rolls,  times, dates, reg, l = extract_access()
    
    return render_template(
        'Access.html',
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        datetoday2=datetoday2
    )
def check_access(identified_person_name):
    """
    Checks if the identified person is a registered user.
    Returns a tuple: (door_state, message)
    """
    registered_users = get_registered_users()  # returns set of IDs

    # identified_person_name format: "Name$ID$"
    if '$' in identified_person_name:
        parts = identified_person_name.split('$')
        if len(parts) >= 2:
            user_id = parts[1]
            if user_id in registered_users:
                access_mail(parts[0])
                return 'unlocked', f'Access granted for {parts[0]}', url_for('files')
            
    intruder_mail()
    # If unknown or not registered
    return 'locked', 'Access Denied' , None

@app.route('/accessbtn', methods=['GET'])
def accessbtn():
    print("[INFO] accessbtn: Starting access capture with liveness detection...")
    
    if len(os.listdir('static/faces')) == 0:
        print("[WARNING] accessbtn: No face images found in database")
        return render_template('Access.html', datetoday2=datetoday2,
                               mess='Database is empty! Register yourself first.')

    print(f"[INFO] accessbtn: Found {len(os.listdir('static/faces'))} users in database")
    
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        print("[INFO] accessbtn: Model not found, training new model...")
        train_model()
    else:
        print("[INFO] accessbtn: Using existing trained model")

    print("[INFO] accessbtn: Opening camera...")
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        print("[ERROR] accessbtn: Failed to open camera")
        names, rolls, times, dates, reg, l = extract_access()
        return render_template('Access.html', names=names, rolls=rolls, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not available.')

    print("[SUCCESS] accessbtn: Camera opened successfully")
    print("[INFO] accessbtn: Starting two-stage verification process...")
    
    ret = True
    frame_count = 0
    
    # Liveness detection variables
    liveness_passed = False
    blink_detected = False
    liveness_timeout = 10  # 10 seconds for liveness check
    liveness_start_time = time.time()  # Track actual time
    
    # Face recognition variables
    j = 1
    flag = -1
    identified_person = "Unknown"
    
    # Stage 1: Liveness Detection
    print("[INFO] accessbtn: STAGE 1 - Liveness Detection (Please blink)")
    
    while ret and not liveness_passed:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Calculate elapsed and remaining time based on REAL time
        elapsed_time = time.time() - liveness_start_time
        remaining_time = max(0, liveness_timeout - elapsed_time)
        
        # Check if timeout exceeded
        if elapsed_time >= liveness_timeout:
            print("[ERROR] accessbtn: Liveness detection timeout")
            break
        
        # Convert frame to RGB for face_recognition library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Check for liveness (blink detection)
        is_live, liveness_info = liveness_detection_from_frame(rgb_frame)
        
        if is_live:
            blink_detected = True
            liveness_passed = True
            print(f"[SUCCESS] accessbtn: Blink detected! Liveness confirmed (EAR: {liveness_info.get('ear', 'N/A')})")
        
        # Draw status on frame
        if blink_detected:
            cv2.putText(frame, 'Liveness: PASSED (Blink detected)', (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Liveness Check: Please BLINK', (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            # Display actual remaining time with 1 decimal place for accuracy
            cv2.putText(frame, f'Time remaining: {remaining_time:.1f}s', 
                       (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(frame, 'Press Space to cancel', (30, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 127, 255), 2, cv2.LINE_AA)
        
        cv2.namedWindow('Access Verification', cv2.WINDOW_NORMAL)
        cv2.imshow('Access Verification', frame)
        
        if cv2.waitKey(1) == 32:  # Space key
            print("[INFO] accessbtn: Space key pressed, canceling...")
            cap.release()
            cv2.destroyAllWindows()
            names, rolls, times, dates, reg, l = extract_access()
            return render_template('Access.html', names=names, rolls=rolls, times=times, l=l,
                                 totalreg=totalreg(), datetoday2=datetoday2, 
                                 mess='Access canceled by user.')
    
    # Check if liveness detection passed
    # Check if liveness detection passed
    if not liveness_passed:
        print("[ERROR] accessbtn: Liveness detection FAILED - No blink detected")
        intruder_mail()
        cap.release()
        cv2.destroyAllWindows()
        return render_template('door.html', door_state='locked', message='Liveness Detection Failed - Access Denied')
   
    
    # Stage 2: Face Recognition (using the SAME camera session)
    print("[INFO] accessbtn: STAGE 2 - Face Recognition")
    print("[INFO] accessbtn: Continuing with same camera session...")
    
    while ret:
        ret, frame = cap.read()
        frame_count += 1
        
        if frame_count % 30 == 0:  # Log every 30 frames (about 1 second)
            print(f"[DEBUG] accessbtn: Processing frame {frame_count}")
        
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            
            try:
                print(f"[DEBUG] accessbtn: Attempting to identify face in frame {frame_count}")
                # Extract HOG features for identification
                try:
                    hog_features = extract_hog_features(face)
                    identified_person = identify_face(hog_features.reshape(1, -1))[0]
                    
                    # Log identification results
                    if identified_person == "Unknown":
                        print(f"[INFO] accessbtn: Face rejected as 'Unknown' - not in registered users list")
                    else:
                        print(f"[INFO] accessbtn: Face identified and verified: {identified_person}")
                        
                except Exception as hog_error:
                    print(f"[ERROR] accessbtn: HOG feature extraction failed: {hog_error}")
                    # Fallback to raw pixels if HOG fails
                    identified_person = identify_face(face.reshape(1, -1))[0]
                
                if '$' in identified_person:
                    identified_person_name = identified_person.split('$')[0]
                    identified_person_id = identified_person.split('$')[1]
                    print(f"[INFO] accessbtn: Identified person: {identified_person_name} (ID: {identified_person_id})")
                else:
                    identified_person_name = "Unknown"
                    identified_person_id = "N/A"
                    print(f"[WARNING] accessbtn: Unknown person identified")

                if flag != identified_person:
                    j = 1
                    flag = identified_person
                    print(f"[INFO] accessbtn: New person detected, resetting counter")

                if j % 20 == 0:
                    print(f"[INFO] accessbtn: Marking access for {identified_person_name}")
                    add_access(identified_person)
            except Exception as e:
                print(f"[ERROR] accessbtn: Error processing identified person: {e}")
                identified_person_name = "Error"
                identified_person_id = "N/A"

            cv2.putText(frame, 'Liveness: PASSED', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Name: {identified_person_name}', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 0, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, f'ID: {identified_person_id}', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 0, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Press Space to close', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 127, 255), 2, cv2.LINE_AA)
            j += 1
        else:
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"[DEBUG] accessbtn: No faces detected in frame {frame_count}")
            j = 1
            flag = -1
            cv2.putText(frame, 'Liveness: PASSED', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'No face detected', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.namedWindow('Access Verification', cv2.WINDOW_NORMAL)
        cv2.imshow('Access Verification', frame)
        
        # Close with Space
        if cv2.waitKey(1) == 32:
            print("[INFO] accessbtn: Space key pressed, closing camera...")
            break

    print("[INFO] accessbtn: Releasing camera resources...")
    cap.release()
    cv2.destroyAllWindows()
    
    door_state, message, redirect_page = check_access(identified_person) 
    print(f"[INFO] accessbtn: Door state: {door_state}, message: {message}")

    if redirect_page:
        # Redirect to files.html automatically
        return redirect(redirect_page)
    else:
        # Show door.html with Access Denied
        return render_template('door.html', door_state=door_state, message=message)


@app.route('/files')
def files():
    # List of secure files (name and display title)
    files = [
        {"title": "Confidential 1", "path": "1.pdf"},
        {"title": "Confidential 2", "path": "2.pdf"},
        {"title": "Confidential 3", "path": "3.pdf"},
        {"title": "Confidential 4", "path": "4.pdf"},
        {"title": "Confidential 5", "path": "5.pdf"},
    ]
    return render_template('files.html', files=files)


@app.route('/adduser')
def add_user():
    return render_template('AddUser.html')

@app.route('/adduserbtn', methods=['GET', 'POST'])
def adduserbtn():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    
    print(f"[INFO] adduserbtn: Starting user registration for {newusername} (ID: {newuserid}")

    # Open camera
    print("[INFO] adduserbtn: Opening camera for face capture...")
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        print("[ERROR] adduserbtn: Failed to open camera")
        return render_template('AddUser.html', mess='Camera not available.')

    print("[SUCCESS] adduserbtn: Camera opened successfully")

    # Create user folder for storing images
    userimagefolder = f'static/faces/{newusername}${newuserid}$'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
        print(f"[INFO] adduserbtn: Created user folder: {userimagefolder}")

    # Check if user already exists in DB
    print(f"[INFO] adduserbtn: Checking if user {newuserid} already exists in database...")
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM student WHERE id = %s", (newuserid,))
    existing_user = cur.fetchone()
    if existing_user:
        print(f"[WARNING] adduserbtn: User {newuserid} already exists in database")
        cap.release()
        cur.close()
        return render_template('AddUser.html', mess='User already exists in database.')

    print(f"[INFO] adduserbtn: User {newuserid} is new, proceeding with face capture")

    images_captured = 0
    max_images = 100
    frame_count = 0

    print(f"[INFO] adduserbtn: Starting face capture loop (target: {max_images} images)")

    while images_captured < max_images:
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            print(f"[WARNING] adduserbtn: Failed to read frame {frame_count}")
            break
            
        faces = extract_faces(frame)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = cv2.resize(frame[y:y+h, x:x+w], (50,50))
                image_path = os.path.join(userimagefolder, f'{images_captured}.jpg')
                cv2.imwrite(image_path, face_img)
                images_captured += 1
                
                if images_captured % 10 == 0:  # Log every 10 images
                    print(f"[INFO] adduserbtn: Captured {images_captured}/{max_images} images")

                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,20), 2)
                
                # Add delay to slow down capture process (0.2 seconds = 20 seconds for 100 images)
                time.sleep(0.2)
        else:
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"[DEBUG] adduserbtn: No faces detected in frame {frame_count}")
                
        cv2.putText(frame, f'Images Captured: {images_captured}/{max_images}', (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,20), 2)
        
        
        # Calculate and display estimated time remaining
        if images_captured > 0:
            remaining_images = max_images - images_captured
            
        
        cv2.namedWindow("Collecting Face Data", cv2.WINDOW_NORMAL)
        cv2.imshow("Collecting Face Data", frame)
        # Close with Space instead of Esc per your preference
        if cv2.waitKey(1) == 32:
            print("[INFO] adduserbtn: Space key pressed, stopping capture...")
            break

    print(f"[INFO] adduserbtn: Face capture completed. Total images: {images_captured}")
    cap.release()
    cv2.destroyAllWindows()

    if images_captured == 0:
        print("[ERROR] adduserbtn: No images captured, cleaning up...")
        shutil.rmtree(userimagefolder)
        return render_template('AddUser.html', mess='Failed to capture valid face images.')

    print(f"[SUCCESS] adduserbtn: Successfully captured {images_captured} face images")

    # Insert new user into MySQL
    print(f"[INFO] adduserbtn: Inserting user {newuserid} into database...")
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("""
        INSERT INTO student (name, id, status)
        VALUES (%s, %s,  'registered')
    """, (newusername, newuserid))
    mysql.connection.commit()
    cur.close()
    print(f"[SUCCESS] adduserbtn: User {newuserid} inserted into database")

    # Retrain model immediately with new user
    print("[INFO] adduserbtn: Retraining model with new user...")
    train_model()

    # Fetch updated registered students
    print("[INFO] adduserbtn: Fetching updated registered students list...")
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM student WHERE status='registered' ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    names = [row['name'] for row in rows]
    rolls = [str(row['id']) for row in rows]
    l = len(rows)
    
    print(f"[INFO] adduserbtn: Found {l} registered students")
    print(f"[SUCCESS] adduserbtn: User registration completed successfully")

    return render_template('RegisterUserList.html',
                           names=names, rolls=rolls, l=l,
                           mess=f'Number of Registered Students: {l}')

@app.route('/accesslist')
def access_list():

    names, rolls,  times, dates, reg, l = extract_access()

    return render_template(
        'AccessList.html',
        names=names, rolls=rolls, times=times, dates=dates, reg=reg, l=l
    )
    
# ========== Flask Search Attendance by Single Date ============
@app.route('/accesslistdate', methods=['GET', 'POST'])
def accesslistdate():

    # Only  date input 
    date_selected = request.form.get('date')

    if not date_selected:
        return render_template('AccessList.html', names=[], rolls=[],  times=[], dates=[], reg=[], l=0,
                               mess="Please select a date")

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # Fetch access for the single date
    cur.execute("""
        SELECT a.name, a.id, a.time,
               COALESCE(s.status, 'Unknown') AS status
        FROM access a
        LEFT JOIN student s ON a.id = s.id
        WHERE DATE(a.time) = %s
        ORDER BY a.time ASC
    """, (date_selected,))
    
    rows = cur.fetchall()
    cur.close()

    mess = f"Access on {date_selected}"

    if not rows:
        return render_template('AccessList.html', names=[], rolls=[],  times=[], dates=[], reg=[], l=0,
                               mess=mess + " No records found.")

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    times = [r['time'].strftime("%H:%M:%S") for r in rows]
    dates = [r['time'].strftime("%Y-%m-%d") for r in rows]
    reg   = [r['status'] for r in rows]
    l     = len(rows)

    return render_template('AccessList.html',
                           names=names, rolls=rolls, 
                           times=times, dates=dates, reg=reg,
                           l=l, mess=f"{mess}: {l} records found")


# ========== Flask Register User List ============
@app.route('/registeruserlist')
def register_user_list():

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cur.execute("SELECT * FROM student WHERE status='registered' ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    names = [row['name'] for row in rows]
    rolls = [row['id'] for row in rows]
    l = len(rows)

    mess = f'Number of Registered Students: {l}' if l else "Database is empty!"
    return render_template('RegisterUserList.html', names=names, rolls=rolls, l=l, mess=mess)

# ========== Flask Delete a User from Registered List ============
@app.route('/deleteregistereduser', methods=['POST'])
def deleteregistereduser():
    

    idx = int(request.form['index'])

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM student WHERE status='registered' ORDER BY id ASC")
    registered = cur.fetchall()

    if idx >= len(registered):
        return render_template('RegisterUserList.html', names=[], rolls=[],  l=0, mess="Invalid user index.")

    user = registered[idx]
    username, userid = user['name'], user['id']

    # Delete face folder
    folder = f'static/faces/{username}${userid}$'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        train_model()

    # Delete from DB
    cur.execute("DELETE FROM student WHERE id = %s and status='registered'", (userid,))
    mysql.connection.commit()
    cur.close()

    # Refresh list
    return redirect(url_for('register_user_list'))



# ======== Flask Login =========
@app.route('/login', methods=['GET', 'POST'])
def login():
    FIXED_PASSCODE = 'redikshya22'

    if request.method == 'POST':
        passcode = request.form.get('passcode')

        if passcode != FIXED_PASSCODE:
            return render_template('LogInForm.html', mess='Wrong passcode')

        # Log login time
        try:
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO admin_table (login_time) VALUES (%s)", (datetime.now(),))
            mysql.connection.commit()
        except Exception as e:
            mysql.connection.rollback()
            print(f"Error inserting login time: {e}")
        finally:
            cur.close()

        # Set session flag
        session['admin_logged_in'] = True
        return redirect(url_for('dashboard'))

    return render_template('LogInForm.html')


# ======== Flask Logout =========
@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template("about.html")

# ======== Flask Dashboard =========
@app.route('/dashboard')
def dashboard():
    # Only allow access if logged in
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))  # redirect to login if not logged in
    return render_template('dashboard.html')


# ======== Flask Home Page =========
@app.route('/')
def home():
    return render_template('HomePage.html')


# ======== Send Passcode Email =========
@app.route('/send-email')
def send_email():
    FIXED_PASSCODE = 'blahblah'
    recipient = 'youremail@gmail.com'  # Fixed admin email

    msg = Message(
        subject='Your Lockify Passcode',
        recipients=[recipient],
        body=f"Hello Admin,\n\nYour passcode for Lockify is: {FIXED_PASSCODE}\n\nRegards,\nLockify Team"
    )

    try:
        mail.send(msg)
        print(f"Successfully sent email to {recipient}")
        return f'''
        <div style="text-align:center; padding:50px;">
            <h1 style="color: green;">Success!</h1>
            <p>Email sent to {recipient}. Check your inbox.</p>
            <p><a href="/login">Go Back to Login</a></p>
        </div>
        '''
    except Exception as e:
        print(f"ERROR: Email failed to send. Details: {e}")
        return f'''
        <div style="text-align:center; padding:50px;">
            <h1 style="color: red;">Error!</h1>
            <p>Email failed to send. Error: {e}</p>
            <p><a href="/login">Go Back to Login</a></p>
        </div>
        '''

# ======== Send intruder Email =========
def intruder_mail():
    """
    Sends an intruder alert email to the admin in a background thread.
    """
    def send_email():
        # Push the Flask app context manually
        with app.app_context():
            recipient = 'youremail@gmail.com'  # Fixed admin email

            msg = Message(
                subject='Alert! Unauthorized Access Attempt Detected',
                recipients=[recipient],
                body=(
                    "Hello Admin,\n\n"
                    "An unauthorized attempt to access Lockify has been detected.\n\n"
                    "Regards,\nLockify Team"
                )
            )

            try:
                mail.send(msg)
                print(f"[INFO] Intruder email successfully sent to {recipient}")
            except Exception as e:
                print(f"[ERROR] Intruder email failed to send. Details: {e}")
                traceback.print_exc()

    # Start the background thread
    threading.Thread(target=send_email, daemon=True).start()


# ======== Send Access Email =========
def access_mail(granted_person):
    """
    Sends an access alert email to the admin in a background thread.
    """
    def send_email():
        # Push the Flask app context manually
        with app.app_context():
            recipient = 'youremail@gmail.com'  # Fixed admin email
            access_time_is = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            msg = Message(
                subject = f"Lockify Access Granted: {granted_person}",
                recipients=[recipient],
                body=(
                    f"Hello Admin,\n\n"
                    f"The access has been granted to {granted_person} .\n"
                    f"Access Time: {access_time_is}\n"
                    f"If this access was unauthorized, please take appropriate action immediately.\n\n"
                    "Regards,\nLockify Team"
                )
            )

            try:
                mail.send(msg)
                print(f"[INFO] Intruder email successfully sent to {recipient}")
            except Exception as e:
                print(f"[ERROR] Intruder email failed to send. Details: {e}")
                traceback.print_exc()

    # Start the background thread
    threading.Thread(target=send_email, daemon=True).start()

# ======== Forgot Passcode =========
@app.route('/forgot_passcode')
def forgot_passcode():
    """Redirects to send_email route to send the fixed passcode to admin"""
    return redirect(url_for('send_email'))


# ======== CNN Face Recognition Routes =========
@app.route('/cnn_try')
def cnn_try():
    """Route to display CNN face recognition page"""
    return render_template('cnn_try.html')

@app.route('/cnn_recognize', methods=['GET'])
def cnn_recognize():
    """Route to execute CNN face recognition"""
    print("[INFO] cnn_recognize: Starting CNN face recognition...")
    
    try:
        # Check if required files exist
        if not os.path.exists('face_recognition_model.h5'):
            print("[ERROR] cnn_recognize: CNN model file not found")
            return render_template('cnn_try.html', mess='CNN model file (face_recognition_model.h5) not found!')
        
        if not os.path.exists('label_encoder.pkl'):
            print("[ERROR] cnn_recognize: Label encoder file not found")
            return render_template('cnn_try.html', mess='Label encoder file (label_encoder.pkl) not found!')
        
        print("[INFO] cnn_recognize: Loading CNN model and label encoder...")
        
        # Import required libraries for CNN
        from tensorflow.keras.models import load_model
        import pickle
        
        # Load CNN model
        model = load_model('face_recognition_model.h5')
        print("[SUCCESS] cnn_recognize: CNN model loaded successfully")
        
        # Load label encoder
        with open('label_encoder.pkl', 'rb') as file:
            label_encoder = pickle.load(file)
        print("[SUCCESS] cnn_recognize: Label encoder loaded successfully")
        
        # Get class names
        class_names = label_encoder.classes_
        print(f"[INFO] cnn_recognize: Class names: {class_names}")
        
        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Define confidence threshold - INCREASED for better unknown detection
        confidence_threshold = 0.80  # Only accept predictions with 90%+ confidence
        
        # Start video capture
        print("[INFO] cnn_recognize: Opening camera...")
        cap = cv2.VideoCapture(0)
        
        if cap is None or not cap.isOpened():
            print("[ERROR] cnn_recognize: Failed to open camera")
            return render_template('cnn_try.html', mess='Camera not available.')
        
        print("[SUCCESS] cnn_recognize: Camera opened successfully")
        
        # Variables to track the best recognition
        recognized_person = "Unknown"
        max_confidence_overall = 0.0
        frame_count = 0
        recognition_count = {}  # Track how many times each person is recognized
        
        print("[INFO] cnn_recognize: Starting face recognition loop...")
        print("[INFO] cnn_recognize: Press 'q' to stop recognition")
        print(f"[INFO] cnn_recognize: Confidence threshold set to {confidence_threshold:.0%}")
        
        while True:
            ret, frame = cap.read()
            frame_count += 1
            
            if not ret:
                print(f"[WARNING] cnn_recognize: Failed to read frame {frame_count}")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            current_frame_label = "No Face Detected"
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Preprocess face for prediction
                face_roi = frame[y:y+h, x:x+w]
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_roi_resized = cv2.resize(face_roi_rgb, (128, 128))
                face_roi_normalized = face_roi_resized.astype('float32') / 255.0
                face_roi_batch = np.expand_dims(face_roi_normalized, axis=0)
                
                # Make prediction
                predictions = model.predict(face_roi_batch, verbose=0)
                max_confidence = np.max(predictions)
                label_index = np.argmax(predictions)
                
                # CRITICAL: Only accept prediction if confidence is above threshold
                if max_confidence >= confidence_threshold:
                    label = label_encoder.inverse_transform([label_index])[0]
                    current_frame_label = label
                    
                    # Track recognition counts
                    if label not in recognition_count:
                        recognition_count[label] = 0
                    recognition_count[label] += 1
                    
                    # Update recognized person if confidence is higher
                    if max_confidence > max_confidence_overall:
                        max_confidence_overall = max_confidence
                        recognized_person = label
                        print(f"[INFO] cnn_recognize: Recognized '{label}' with confidence {max_confidence:.2%}")
                    
                    # Display recognized label with green color
                    cv2.putText(frame, f'{label}', 
                               (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f'Confidence: {max_confidence:.2%}', 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Low confidence - mark as Unknown
                    label = "Unknown"
                    current_frame_label = "Unknown"
                    print(f"[INFO] cnn_recognize: Low confidence ({max_confidence:.2%}) - Marked as Unknown")
                    
                    # Display Unknown with red color
                    cv2.putText(frame, 'Unknown', 
                               (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f'Confidence: {max_confidence:.2%} (Too Low)', 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display instructions and current status
            cv2.putText(frame, 'Press Q to stop and return home', (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f'Current: {current_frame_label}', (30, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Show frame
            cv2.namedWindow('CNN Face Recognition', cv2.WINDOW_NORMAL)
            cv2.imshow('CNN Face Recognition', frame)
            
            # Check for 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] cnn_recognize: 'q' key pressed, stopping recognition...")
                break
        
        # Release resources
        print("[INFO] cnn_recognize: Releasing camera resources...")
        cap.release()
        cv2.destroyAllWindows()
        
        # Final decision logic: Use the most frequently recognized person
        # Only if they were recognized enough times with high confidence
        if recognition_count:
            # Get the person recognized most frequently
            most_recognized = max(recognition_count, key=recognition_count.get)
            recognition_times = recognition_count[most_recognized]
            
            print(f"[INFO] cnn_recognize: Recognition counts: {recognition_count}")
            
            # Only accept if recognized at least 5 times (to avoid false positives)
            if recognition_times >= 4:
                recognized_person = most_recognized
                print(f"[SUCCESS] cnn_recognize: Final result: {recognized_person} (recognized {recognition_times} times)")
            else:
                recognized_person = "Unknown"
                print(f"[INFO] cnn_recognize: Insufficient recognitions ({recognition_times} < 5), marking as Unknown")
        else:
            recognized_person = "Unknown"
            print(f"[INFO] cnn_recognize: No valid recognitions above threshold, marking as Unknown")
        
        # Format confidence for display
        confidence_display = f"{max_confidence_overall:.2%}" if max_confidence_overall > 0 else "N/A"
        
        print(f"[SUCCESS] cnn_recognize: Recognition completed. Final result: {recognized_person} (max confidence: {confidence_display})")
        print("[INFO] cnn_recognize: Redirecting to homepage...")
        
        # Redirect to homepage after recognition completes
        return redirect(url_for('home'))
        
    except Exception as e:
        print(f"[ERROR] cnn_recognize: An error occurred - {e}")
        traceback.print_exc()
        return render_template('cnn_try.html', mess=f'Error during CNN recognition: {str(e)}')
# Main Function
if __name__ == '__main__':
    print("=" * 60)
    print(" Starting Face Recognition Lockify ")
    print("=" * 60)
    print(f"[INFO] Main: Current date: {datetoday}")
    print(f"[INFO] Main: Current date (formatted): {datetoday2}")
    print(f"[INFO] Main: Face detector loaded: {face_detector is not None}")
    print(f"[INFO] Main: Static faces directory: {'static/faces' in os.listdir('.')}")
    print(f"[INFO] Main: KNN model will be trained on-demand when needed")
    print("=" * 60)
    
    # KNN model is trained on-demand when needed
    app.run(port=5001, debug=True)
