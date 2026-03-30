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
import joblib
from flask_mysqldb import MySQL
import MySQLdb.cursors
import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask_mail import Mail, Message
import threading
import traceback
from liveness import liveness_detection
import time
from model_evaluator import ModelEvaluator
import aug
from ensemble_knn import EnsembleKNN

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
app.config['MAIL_USERNAME'] = '021bscit017@sxc.edu.np'

app.config['MAIL_PASSWORD'] = 'mswj vnmp koba hsxc'

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
        
        hog_features = hog(
            gray_face,
            orientations=8,           
            pixels_per_cell=(8, 8),   
            cells_per_block=(2, 2),   
            block_norm='L2-Hys',      
            visualize=False,          
            feature_vector=True       
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
                resized_face = cv2.resize(img, (56, 56))
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
    print(f"[INFO] train_model: Training Ensemble KNN model on {len(X_train)} samples...")
    knn = EnsembleKNN(k_values=[1,3, 5])  
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
    
    # Clean up old results (keep only latest 2)
    evaluator.cleanup_old_results(keep_latest=2)
    
    print(f"[SUCCESS] train_model: Model evaluation completed")
    
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
        cur.execute("SELECT id FROM uuser WHERE status='registered'")
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
    cur.execute("SELECT id FROM uuser WHERE status='registered'")
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
    SELECT a.name, a.id, a.access_date,
           COALESCE(s.status, 'Unknown') AS status,
           COALESCE(s.department, 'N/A') AS department
    FROM access a
    LEFT JOIN uuser s ON a.id = s.id
    WHERE DATE(a.access_date) = %s
    ORDER BY a.access_date ASC
    """

    cur.execute(query, (datetoday_mysql,))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return [], [], datetoday, [], [], 0

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    reg = [r['status'] for r in rows]
    departments = [r['department'] for r in rows]
    l = len(rows)

    return names, rolls, datetoday, reg, departments, l

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
        current_datetime = datetime.now().strftime("%Y-%m-%d")
        
        print(f"[INFO] add_access: User: {username}, ID: {userid}")
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("""
            SELECT * FROM access 
            WHERE id=%s AND DATE(access_date)=%s
    """, (userid, datetime.now().strftime("%Y-%m-%d")))
        already = cur.fetchone()

        if already:
            cur.close()
            return 

        # Insert new access with full date
        print(f"[INFO] add_access: Inserting new access record for user {userid}")
        cur.execute("""
            INSERT INTO access (id, name, access_date)
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
    names, rolls,  dates, reg, departments, l = extract_access()
    
    return render_template(
        'Access.html',
        names=names,
        rolls=rolls,
        departments=departments,
        l=l,
        datetoday2=datetoday2
    )


def check_access(identified_person_name):
    """
    Checks if the identified person is a registered user.
    Returns a tuple: (door_state, message, redirect_url, department)
    """
    try:
        registered_users = get_registered_users()  # returns set of IDs

        # identified_person_name format: "Name$ID$"
        if '$' in identified_person_name:
            parts = identified_person_name.split('$')
            if len(parts) >= 2:
                user_id = parts[1]
                user_name = parts[0]

                if user_id in registered_users:
                    # Fetch department from uuser table
                    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                    cur.execute("SELECT department FROM uuser WHERE id=%s", (user_id,))
                    result = cur.fetchone()
                    cur.close()

                    if result:
                        user_department = result['department']
                    else:
                        user_department = None  # fallback

                    access_mail(user_name)
                    # Return department along with other info
                    return 'unlocked', f'Access granted for {user_name}', url_for('files'), user_department

        # If unknown or not registered
        intruder_mail()
        return 'locked', 'Access Denied', None, None

    except Exception as e:
        print(f"[ERROR] check_access: {e}")
        return 'locked', 'Access Denied', None, None


@app.route('/accessbtn', methods=['GET'])
def accessbtn():
    print("[INFO] accessbtn: Starting access capture with liveness detection...")
    
    if len(os.listdir('static/faces')) == 0:
        print("[WARNING] accessbtn: No face images found in database")
        return render_template('Access.html', datetoday2=datetoday2,
                               mess='Database is empty! Register yourself first.')

    print(f"[INFO] accessbtn: Found {len(os.listdir('static/faces'))} users folders")
    
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        print("[INFO] accessbtn: Model not found, training new model...")
        train_model()
    else:
        print("[INFO] accessbtn: Using existing trained model")

    print("[INFO] accessbtn: Opening camera...")
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        print("[ERROR] accessbtn: Failed to open camera")
        names, rolls,  dates, reg, departments, l = extract_access()
        return render_template('Access.html', names=names, rolls=rolls,departments=departments,  l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not available.')

    print("[SUCCESS] accessbtn: Camera opened successfully")
    print("[INFO] accessbtn: Starting two-stage verification process...")
    
    ret = True
    frame_count = 0
    
    # Liveness detection variables
    liveness_passed = False
    blink_detected = False
    liveness_timeout = 15  # 10 seconds for liveness check
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
        is_live, liveness_info = liveness_detection(rgb_frame)
        
        if is_live:
            blink_detected = True
            liveness_passed = True
            print(f"[SUCCESS] accessbtn: Blink detected! Liveness confirmed (EAR: {liveness_info.get('ear', 'N/A')})")
        
        # Draw status on frame
        if blink_detected:
            cv2.putText(frame, 'Liveness: PASSED (Blink detected)', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Liveness Check: Please BLINK', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            # Display actual remaining time with 1 decimal place for accuracy
            cv2.putText(frame, f'Time remaining: {remaining_time:.1f}s', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(frame, 'Press Space to cancel', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 127, 255), 2, cv2.LINE_AA)
        
        cv2.namedWindow('Access Verification', cv2.WINDOW_NORMAL)
        cv2.imshow('Access Verification', frame)
        
        if cv2.waitKey(1) == 32:  # Space key
            print("[INFO] accessbtn: Space key pressed, canceling...")
            cap.release()
            cv2.destroyAllWindows()
            names, rolls,  dates, reg, departments, l = extract_access()
            return render_template('Access.html', names=names, rolls=rolls, departments=departments, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Access canceled by user.')
    
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
        
        if frame_count % 30 == 0:  
            print(f"[DEBUG] accessbtn: Processing frame {frame_count}")
        
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (56, 56))
            
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

                if j % 30 == 0:
                    print(f"[INFO] accessbtn: Marking access for {identified_person_name}")
                    add_access(identified_person)
            except Exception as e:
                print(f"[ERROR] accessbtn: Error processing identified person: {e}")
                identified_person_name = "Error"
                identified_person_id = "N/A"

            cv2.putText(frame, 'Liveness: PASSED', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Name: {identified_person_name}', (30, 70), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, f'ID: {identified_person_id}', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Press Space to close', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 127, 255), 2, cv2.LINE_AA)
            j += 1
        else:
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"[DEBUG] accessbtn: No faces detected in frame {frame_count}")
            j = 1
            flag = -1
            cv2.putText(frame, 'Liveness: PASSED', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'No face detected', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.namedWindow('Access Verification', cv2.WINDOW_NORMAL)
        cv2.imshow('Access Verification', frame)
        
        # Close with Space
        if cv2.waitKey(1) == 32:
            print("[INFO] accessbtn: Space key pressed, closing camera...")
            break

    print("[INFO] accessbtn: Releasing camera resources...")
    cap.release()
    cv2.destroyAllWindows()
    
    door_state, message, redirect_page, user_department = check_access(identified_person)
    print(f"[INFO] accessbtn: Door state: {door_state}, message: {message}")

    if door_state == 'unlocked' and user_department:
    # Redirect to files page with department info as query parameter
        return redirect(url_for('files', user_department=user_department))
    else:
    # Show door.html with Access Denied
        return render_template('door.html', door_state=door_state, message=message)


@app.route('/files', methods=['GET', 'POST'])
def files():
    user_department = request.args.get('user_department') 

    if not user_department:
        return "Department info missing. Access denied.", 403

    files_list = []
    message = ''

    if request.method == 'POST':
        search_query = request.form.get('search').strip()
        if search_query:
            cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cur.execute(
                "SELECT * FROM files WHERE file_name LIKE %s",
                ('%' + search_query + '%',)
            )
            results = cur.fetchall()
            cur.close()

            if results:
                for file in results:
                    if file['department'] == user_department:
                        files_list.append(file)
                    else:
                        message = "File not accessible outside your department."
            else:
                message = "No files found with that name."

    return render_template('files.html', files=files_list, message=message)


@app.route('/adduser')
def add_user():
    return render_template('AddUser.html')

@app.route('/adduserbtn', methods=['GET', 'POST'])
def adduserbtn():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    newdepartment = request.form['department']  

    
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
    cur.execute("SELECT * FROM uuser WHERE id = %s", (newuserid,))
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
                face_img = cv2.resize(frame[y:y+h, x:x+w], (56,56))
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
    INSERT INTO uuser (name, id, department, status)
    VALUES (%s, %s, %s, 'registered')
    """, (newusername, newuserid, newdepartment))

    mysql.connection.commit()
    cur.close()
    print(f"[SUCCESS] adduserbtn: User {newuserid} inserted into database")

    # Run augmentation on captured faces before training
    print("[INFO] adduserbtn: Running data augmentation before training...")
    aug.main()
    print("[SUCCESS] adduserbtn: Data augmentation completed.")


    # Retrain model immediately with new user
    print("[INFO] adduserbtn: Retraining model with new user...")
    train_model()

    # Fetch updated registered users
    print("[INFO] adduserbtn: Fetching updated registered users list...")
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM uuser WHERE status='registered' ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    names = [row['name'] for row in rows]
    rolls = [str(row['id']) for row in rows]
    departments = [row['department'] for row in rows]  
    l = len(rows)
    
    print(f"[INFO] adduserbtn: Found {l} registered users")
    print(f"[SUCCESS] adduserbtn: User registration completed successfully")

    return render_template('RegisterUserList.html',
                       names=names, rolls=rolls, departments=departments,
                       l=l, mess=f'Number of Registered users: {l}')


# ========== Access list ============
@app.route('/accesslist')
def access_list():
    names, rolls, dates, reg, departments, l = extract_access()

    return render_template(
        'AccessList.html',
        names=names, 
        rolls=rolls, 
        dates=dates, 
        reg=reg, 
        departments=departments,
        l=l
    )
    
# ========== Flask Search Access by Single Date ============
@app.route('/accesslistdate', methods=['GET', 'POST'])
def accesslistdate():
    date_selected = request.form.get('date')

    if not date_selected:
        return render_template('AccessList.html', names=[], rolls=[], dates=[], reg=[], departments=[], l=0,
                               mess="Please select a date")

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cur.execute("""
        SELECT a.name, a.id, a.access_date,
               COALESCE(s.status, 'Unknown') AS status,
               COALESCE(s.department, 'N/A') AS department
        FROM access a
        LEFT JOIN uuser s ON a.id = s.id
        WHERE DATE(a.access_date) = %s
        ORDER BY a.access_date ASC
    """, (date_selected,))
    
    rows = cur.fetchall()
    cur.close()

    mess = f"Access on {date_selected}"

    if not rows:
        return render_template('AccessList.html', names=[], rolls=[], dates=[], reg=[], departments=[], l=0,
                               mess=mess + " No records found.")

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    dates = [r['access_date'].strftime("%Y-%m-%d") for r in rows]
    reg = [r['status'] for r in rows]
    departments = [r['department'] for r in rows]
    l = len(rows)

    return render_template('AccessList.html',
                           names=names, 
                           rolls=rolls, 
                           dates=dates, 
                           reg=reg,
                           departments=departments,
                           l=l, 
                           mess=f"{mess}: {l} records found")


# ========== Flask Register User List ============
@app.route('/registeruserlist')
def register_user_list():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cur.execute("SELECT * FROM uuser WHERE status='registered' ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    names = [row['name'] for row in rows]
    rolls = [row['id'] for row in rows]
    departments = [row['department'] for row in rows]
    l = len(rows)

    mess = f'Number of Registered users: {l}' if l else "Database is empty!"
    return render_template('RegisterUserList.html', 
                          names=names, 
                          rolls=rolls, 
                          departments=departments,
                          l=l, 
                          mess=mess)


# ========== Flask Delete a User from Registered List ============
@app.route('/deleteregistereduser', methods=['POST'])
def deleteregistereduser():
    

    idx = int(request.form['index'])

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM uuser WHERE status='registered' ORDER BY id ASC")
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
    cur.execute("DELETE FROM uuser WHERE id = %s and status='registered'", (userid,))
    mysql.connection.commit()
    cur.close()

    # Refresh list
    return redirect(url_for('register_user_list'))

# ======== Change Passcode =========
@app.route('/change_pw', methods=['GET', 'POST'])
def change_pw():
    mess = ""

    if request.method == 'POST':
        current_pass = request.form['current_passcode']
        new_pass = request.form['new_passcode']

        try:
            cur = mysql.connection.cursor()

            # 1. Retrieve the current passcode from the database
            cur.execute("SELECT passcode FROM admin_passcode ORDER BY passcode_id DESC LIMIT 1")
            result = cur.fetchone()

            if result:
                db_passcode = result[0]

                # 2. Compare entered current passcode with stored one
                if current_pass == db_passcode:
                    # 3. Delete old passcode
                    cur.execute("DELETE FROM admin_passcode")

                    # 4. Insert the new passcode
                    cur.execute("INSERT INTO admin_passcode (passcode) VALUES (%s)", (new_pass,))
                    mysql.connection.commit()

                    mess = "Passcode updated successfully."
                else:
                    mess = "Incorrect current passcode."
            else:
                mess = "No passcode found in the database."

        except Exception as e:
            mysql.connection.rollback()
            print("Error updating passcode:", e)
            mess = "Error occurred while updating passcode."

        finally:
            cur.close()

    return render_template('change_pw.html', mess=mess)


# ======== Flask Login =========
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        passcode = request.form.get('passcode')

        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT passcode FROM admin_passcode ORDER BY passcode_id DESC LIMIT 1")
            result = cur.fetchone()
            stored_passcode = result[0] if result else None
        except Exception as e:
            print("Error retrieving passcode:", e)
            stored_passcode = None
        finally:
            cur.close()
        print("Entered passcode:", passcode)
        print("Stored passcode:", stored_passcode)

        # Compare the entered passcode with stored one
        if not stored_passcode or passcode != stored_passcode:
            return render_template('LogInForm.html', mess='Wrong passcode')
        

        # Set session flag
        session['admin_logged_in'] = True
        return redirect(url_for('dashboard'))

    return render_template('LogInForm.html')

# ======== Flask Logout =========
@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('home'))

# ======== Flask about =========
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
    try:
        cur = mysql.connection.cursor()

        # Fetch all admin emails
        cur.execute("SELECT email FROM admin_table")
        result = cur.fetchall()

        if not result:
            return '''
            <div style="text-align:center; padding:50px;">
                <h1 style="color: red;">Error!</h1>
                <p>No admin emails found in the database.</p>
                <p><a href="/login">Go Back to Login</a></p>
            </div>
            '''

        # Convert list of tuples to list of email strings
        recipient = [row[0] for row in result]

        # Fetch the latest passcode from the database
        cur.execute("SELECT passcode FROM admin_passcode ORDER BY passcode_id DESC LIMIT 1")
        passcode_result = cur.fetchone()
        cur.close()

        if not passcode_result:
            return '''
            <div style="text-align:center; padding:50px;">
                <h1 style="color: red;">Error!</h1>
                <p>No passcode found in the database.</p>
                <p><a href="/login">Go Back to Login</a></p>
            </div>
            '''

        db_passcode = passcode_result[0]

        # Create the email message
        msg = Message(
            subject='Your Lockify Passcode',
            recipients=recipient,
            body=f"Hello Admin,\n\nYour current Lockify passcode is: {db_passcode}\n\nRegards,\nLockify Team"
        )

        # Send the email
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
            try:
                cur = mysql.connection.cursor()

                # Fetch all admin emails
                cur.execute("SELECT email FROM admin_table")
                result = cur.fetchall()
                cur.close()

                if not result:
                    print("[ERROR] No admin emails found in admin_table.")
                    return

                # Convert list of tuples to list of email strings
                recipient = [row[0] for row in result]
                access_time_is = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                msg = Message(
                    subject='Alert! Unauthorized Access Attempt Detected',
                    recipients=recipient,
                    body=(
                        f"Hello Admin,\n\n"
                        f"An unauthorized attempt to access Lockify has been detected.\n\n"
                        f"Access Time: {access_time_is}\n"
                        f"Regards,\nLockify Team"
                    )
                )

                mail.send(msg)
                print(f"[INFO] Intruder alert email successfully sent to {recipient}")

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
            try:
                cur = mysql.connection.cursor()

                # Fetch all admin emails
                cur.execute("SELECT email FROM admin_table")
                result = cur.fetchall()
                cur.close()

                if not result:
                    print("[ERROR] No admin emails found in admin_table.")
                    return

                # Convert list of tuples to list of email strings
                recipient = [row[0] for row in result]
                access_time_is = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                msg = Message(
                    subject=f"Lockify Access Granted: {granted_person}",
                    recipients=recipient,
                    body=(
                        f"Hello Admin,\n\n"
                        f"Access has been granted to {granted_person}.\n"
                        f"Access Time: {access_time_is}\n"
                        f"If this access was unauthorized, please take appropriate action immediately.\n\n"
                        f"Regards,\nLockify Team"
                    )
                )

                mail.send(msg)
                print(f"[INFO] Access email successfully sent to {recipient}")

            except Exception as e:
                print(f"[ERROR] Access email failed to send. Details: {e}")
                traceback.print_exc()

    # Start the background thread
    threading.Thread(target=send_email, daemon=True).start()


# ======== Forgot Passcode =========
@app.route('/forgot_passcode')
def forgot_passcode():
    """Redirects to send_email route to send the fixed passcode to admin"""
    return redirect(url_for('send_email'))

# ======== View Admins =========
@app.route('/view_admins')
def view_admins():
    mess = None
    admins = []
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT email FROM admin_table ORDER BY id DESC")  # select only emails
        result = cur.fetchall()  # returns list of tuples like [('admin3@example.com',), ('admin2@example.com',)...]
        admins = [row[0] for row in result]  # extract emails only
        cur.close()
    except Exception as e:
        mess = f"Error fetching admins: {e}"

    return render_template('admins.html', admins=admins, mess=mess)


# ======== Add Admin =========
@app.route('/add_admin', methods=['GET', 'POST'])
def add_admin():
    mess = None

    try:
        cur = mysql.connection.cursor()

        # Fetch the latest passcode to email
        cur.execute("SELECT passcode FROM admin_passcode ORDER BY passcode_id DESC LIMIT 1")
        passcode_result = cur.fetchone()
        current_passcode = passcode_result[0] if passcode_result else None

        if request.method == 'POST':
            email = request.form['email'].strip()

            # Check if email already exists
            cur.execute("SELECT * FROM admin_table WHERE email = %s", (email,))
            existing = cur.fetchone()

            if existing:
                mess = f"Admin with email {email} already exists."
            else:
                # Insert new admin email
                cur.execute("INSERT INTO admin_table (email) VALUES (%s)", (email,))
                mysql.connection.commit()

                # Send the current passcode to the new admin via email
                if current_passcode:
                    msg = Message(
                        subject="Lockify Admin Access",
                        recipients=[email],
                        body=f"Hello Admin,\n\nYou have been added to Lockify.\nCurrent passcode: {current_passcode}\n\nRegards,\nLockify Team"
                    )
                    mail.send(msg)

                mess = f"Admin added successfully! Passcode has been emailed to {email}."

        cur.close()

    except Exception as e:
        mess = f"Error adding admin: {e}"

    return render_template('add_admin.html', mess=mess)


# ======== CNN Face Recognition Routes =========
@app.route('/cnn_try')
def cnn_try():
    """Route to display CNN face recognition page"""
    return render_template('cnn_try.html')

@app.route('/cnn_recognize', methods=['GET'])
def cnn_recognize():
    """Route to execute CNN face recognition with liveness detection"""
    print("[INFO] cnn_recognize: Starting CNN face recognition with liveness detection...")
    
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
        
        # Define confidence threshold
        confidence_threshold = 0.90
        
        # Start video capture
        print("[INFO] cnn_recognize: Opening camera...")
        cap = cv2.VideoCapture(0)
        
        if cap is None or not cap.isOpened():
            print("[ERROR] cnn_recognize: Failed to open camera")
            return render_template('cnn_try.html', mess='Camera not available.')
        
        print("[SUCCESS] cnn_recognize: Camera opened successfully")
        
        # Variables for recognition
        recognized_person = "Unknown"
        max_confidence_overall = 0.0
        frame_count = 0
        recognition_count = {}
        
        # Liveness detection variables
        liveness_passed = False
        blink_detected = False
        liveness_timeout = 15  # 10 seconds for liveness check
        liveness_start_time = time.time()
        
        # Stage 1: Liveness Detection
        print("[INFO] cnn_recognize: STAGE 1 - Liveness Detection (Please blink)")
        
        while not liveness_passed:
            ret, frame = cap.read()
            
            if not ret:
                print("[WARNING] cnn_recognize: Failed to read frame")
                break
            
            # Calculate elapsed and remaining time based on REAL time
            elapsed_time = time.time() - liveness_start_time
            remaining_time = max(0, liveness_timeout - elapsed_time)
            
            # Check if timeout exceeded
            if elapsed_time >= liveness_timeout:
                print("[ERROR] cnn_recognize: Liveness detection timeout")
                break
            
            # Convert frame to RGB for face_recognition library
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Check for liveness (blink detection)
            is_live, liveness_info = liveness_detection(rgb_frame)
            
            if is_live:
                blink_detected = True
                liveness_passed = True
                ear_value = liveness_info.get('ear', 'N/A')
                print(f"[SUCCESS] cnn_recognize: Blink detected! Liveness confirmed (EAR: {ear_value})")
                print(f"[INFO] cnn_recognize: Liveness check completed in {elapsed_time:.2f} seconds")
            
            # Draw status on frame
            if blink_detected:
                cv2.putText(frame, 'Liveness: PASSED (Blink detected)', (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Liveness Check: Please BLINK', (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Time remaining: {remaining_time:.1f}s', 
                           (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(frame, 'Press Q to cancel', (30, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 127, 255), 2, cv2.LINE_AA)
            
            cv2.namedWindow('CNN Face Recognition', cv2.WINDOW_NORMAL)
            cv2.imshow('CNN Face Recognition', frame)
            
            # Check for 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] cnn_recognize: 'q' key pressed, canceling...")
                cap.release()
                cv2.destroyAllWindows()
                return redirect(url_for('home'))
        
        # Check if liveness detection passed
        if not liveness_passed:
            print("[ERROR] cnn_recognize: Liveness detection FAILED - No blink detected")
            intruder_mail()
            cap.release()
            cv2.destroyAllWindows()
            return redirect(url_for('home'))
        
        # Stage 2: CNN Face Recognition (using the SAME camera session)
        print("[INFO] cnn_recognize: STAGE 2 - CNN Face Recognition")
        print("[INFO] cnn_recognize: Continuing with same camera session...")
        print("[INFO] cnn_recognize: Press 'q' to stop recognition")
        
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
                    print(f"[INFO] cnn_recognize: Low confidence ({max_confidence:.2%}) - Marked as Unknown")
                    
                    # Display Unknown with red color
                    cv2.putText(frame, 'Unknown', 
                               (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f'Confidence: {max_confidence:.2%} (Too Low)', 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display instructions and liveness status
            cv2.putText(frame, 'Liveness: PASSED', (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Press Q to stop and return home', (30, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
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
        
        # Final decision logic
        if recognition_count:
            # Get the person recognized most frequently
            most_recognized = max(recognition_count, key=recognition_count.get)
            recognition_times = recognition_count[most_recognized]
            
            print(f"[INFO] cnn_recognize: Recognition counts: {recognition_count}")
            
            # Only accept if recognized at least 5 times
            if recognition_times >= 5:
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
    
    app.run(port=5001, debug=True)