from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from collections import deque
import time
import logging

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)

# Initialize camera with better settings
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

# Buffers for smoothing and history
emotion_buffer = deque(maxlen=15)
neutro_history = deque(maxlen=100)
emotion_timeline = deque(maxlen=100)

# Track session statistics
session_stats = {
    'total_frames': 0,
    'emotion_counts': {},
    'start_time': time.time()
}

# Load cascades globally
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Try to import DeepFace for enhanced accuracy
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully - Using Deep Learning model")
except ImportError:
    print("⚠️  DeepFace not available - Using enhanced Haar Cascade model")
    print("💡 For better accuracy, install: pip install deepface tensorflow")

def convert_to_native_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj

def calculate_neutrosophic_values(emotions_dict):
    """
    Advanced Neutrosophic Logic Calculation
    T (Truth): Normalized confidence in detected emotion
    I (Indeterminacy): Uncertainty based on emotion distribution
    F (Falsity): Complement of truth (contradiction degree)
    """
    scores = list(emotions_dict.values())
    max_score = max(scores)
    min_score = min(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
   
    # Calculate entropy for indeterminacy
    scores_norm = np.array(scores) / 100.0
    entropy = -np.sum(scores_norm * np.log2(scores_norm + 1e-10))
    max_entropy = np.log2(len(scores))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
   
    neutro_emotions = {}
    for emotion, score in emotions_dict.items():
        T = score / 100.0
        closeness_to_mean = 1 - abs(score - mean_score) / 50.0
        closeness_to_mean = max(0, min(1, closeness_to_mean))
        variance_factor = min(std_score / 25.0, 1.0)
        entropy_factor = normalized_entropy
        I = (closeness_to_mean * 0.4 + (1 - variance_factor) * 0.3 + entropy_factor * 0.3)
        I = max(0, min(1, I))
        F = 1 - T
        total = T + I + F
        T_norm = T / total if total > 0 else 0.33
        I_norm = I / total if total > 0 else 0.33
        F_norm = F / total if total > 0 else 0.33
        certainty_index = T * (1 - I)
        ambiguity_index = I * (1 - abs(T - F))
       
        neutro_emotions[emotion] = {
            'score': float(round(score, 2)),
            'T': float(round(T, 4)),
            'I': float(round(I, 4)),
            'F': float(round(F, 4)),
            'T_norm': float(round(T_norm, 4)),
            'I_norm': float(round(I_norm, 4)),
            'F_norm': float(round(F_norm, 4)),
            'certainty': float(round(certainty_index, 4)),
            'ambiguity': float(round(ambiguity_index, 4)),
            'entropy': float(round(entropy, 4))
        }
   
    return neutro_emotions

def deepface_emotion_detector(frame):
    """
    Deep Learning emotion detection using DeepFace (FER model)
    95%+ accuracy on standard datasets
    """
    try:
        # Analyze with DeepFace
        results = DeepFace.analyze(
            frame, 
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        
        # Handle list or dict response
        if isinstance(results, list):
            results = results[0]
        
        # Extract emotions
        emotions_raw = results.get('emotion', {})
        dominant = results.get('dominant_emotion', 'neutral')
        
        # Map to standard 7 emotions
        emotion_mapping = {
            'angry': 'angry',
            'disgust': 'disgust', 
            'fear': 'fear',
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
        
        emotions = {}
        for key, value in emotions_raw.items():
            mapped_key = emotion_mapping.get(key, 'neutral')
            if mapped_key in emotions:
                emotions[mapped_key] += value
            else:
                emotions[mapped_key] = value
        
        # Ensure all 7 emotions exist
        for emotion in ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']:
            if emotion not in emotions:
                emotions[emotion] = 0.1
        
        # Normalize to 100 and convert to float
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: float((v/total)*100) for k, v in emotions.items()}
        
        return emotions, emotion_mapping.get(dominant, 'neutral')
        
    except Exception as e:
        print(f"DeepFace error: {e}")
        # Fallback to advanced detector
        return advanced_haar_emotion_detector(frame)

def advanced_haar_emotion_detector(frame):
    """
    ULTRA-ENHANCED Haar Cascade emotion detection
    Multi-feature fusion with weighted scoring
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better feature detection
    gray = cv2.equalizeHist(gray)
    
    # Detect faces with multiple scales
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5,
        minSize=(80, 80)
    )
    
    if len(faces) == 0:
        return {
            'happy': 5.0, 'sad': 5.0, 'angry': 5.0,
            'surprise': 5.0, 'fear': 5.0, 'disgust': 5.0, 'neutral': 70.0
        }, 'neutral'
    
    # Get largest face
    (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    
    # Initialize emotion scores
    scores = {
        'happy': 0.0, 'sad': 0.0, 'angry': 0.0,
        'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0, 'neutral': 20.0
    }
    
    # ===== FEATURE EXTRACTION =====
    
    # 1. EYE DETECTION (with better parameters)
    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(20, 20)
    )
    num_eyes = len(eyes)
    
    # Calculate eye aspect ratio if 2 eyes detected
    eye_openness = 0
    if num_eyes >= 2:
        eye_heights = [e[3] for e in eyes[:2]]
        eye_widths = [e[2] for e in eyes[:2]]
        eye_openness = np.mean(eye_heights) / (np.mean(eye_widths) + 1e-5)
    
    # 2. SMILE DETECTION (multiple parameter sets)
    smiles_strong = smile_cascade.detectMultiScale(roi_gray, 1.7, 22, minSize=(25, 25))
    smiles_weak = smile_cascade.detectMultiScale(roi_gray, 1.8, 18, minSize=(20, 20))
    smile_confidence = len(smiles_strong) * 2 + len(smiles_weak)
    
    # 3. FACIAL REGION ANALYSIS
    h_face, w_face = roi_gray.shape
    
    # Divide face into regions
    forehead = roi_gray[0:int(h_face*0.35), :]
    eyes_region = roi_gray[int(h_face*0.25):int(h_face*0.55), :]
    nose_region = roi_gray[int(h_face*0.4):int(h_face*0.7), :]
    mouth_region = roi_gray[int(h_face*0.55):int(h_face*0.9), :]
    
    # Calculate regional statistics
    forehead_var = float(np.var(forehead)) if forehead.size > 0 else 0.0
    eyes_var = float(np.var(eyes_region)) if eyes_region.size > 0 else 0.0
    nose_var = float(np.var(nose_region)) if nose_region.size > 0 else 0.0
    mouth_var = float(np.var(mouth_region)) if mouth_region.size > 0 else 0.0
    
    forehead_mean = float(np.mean(forehead)) if forehead.size > 0 else 0.0
    mouth_mean = float(np.mean(mouth_region)) if mouth_region.size > 0 else 0.0
    
    # 4. EDGE DETECTION (mouth activity)
    mouth_edges = cv2.Canny(mouth_region, 50, 150) if mouth_region.size > 0 else np.array([])
    edge_density = float(np.sum(mouth_edges) / (mouth_edges.size + 1e-5))
    
    # 5. FACIAL CONTOURS
    contours_mouth, _ = cv2.findContours(
        cv2.threshold(mouth_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    ) if mouth_region.size > 0 else ([], None)
    num_contours = len(contours_mouth)
    
    # 6. SYMMETRY ANALYSIS
    left_half = roi_gray[:, 0:int(w_face*0.5)]
    right_half = cv2.flip(roi_gray[:, int(w_face*0.5):w_face], 1)
    min_width = min(left_half.shape[1], right_half.shape[1])
    asymmetry = float(np.mean(np.abs(left_half[:, :min_width].astype(float) - 
                                 right_half[:, :min_width].astype(float))))
    
    # 7. GRADIENT ANALYSIS (facial tension)
    sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
    
    # ===== EMOTION SCORING RULES =====
    
    # HAPPY: Strong smile + bright features + high mouth activity
    if smile_confidence >= 3:
        scores['happy'] += 50
    elif smile_confidence >= 1:
        scores['happy'] += 30
    
    if num_eyes >= 2 and eye_openness > 0.4:
        scores['happy'] += 20
    
    if mouth_var > 850:
        scores['happy'] += 25
    
    if edge_density > 0.05:
        scores['happy'] += 20
    
    if mouth_mean < 100:
        scores['happy'] += 15
    
    # SAD: Low facial activity + downturned features + closed appearance
    if smile_confidence == 0:
        scores['sad'] += 25
    
    if mouth_var < 500:
        scores['sad'] += 30
    
    if edge_density < 0.03:
        scores['sad'] += 25
    
    if num_eyes <= 1 or eye_openness < 0.35:
        scores['sad'] += 20
    
    if forehead_var < 700 and mouth_var < 600:
        scores['sad'] += 20
    
    # ANGRY: Furrowed brow + tension + asymmetry + compressed features
    if forehead_var > 1100:
        scores['angry'] += 35
    
    if gradient_magnitude > 25:
        scores['angry'] += 30
    
    if asymmetry > 20:
        scores['angry'] += 25
    
    if eyes_var > 900 and smile_confidence == 0:
        scores['angry'] += 25
    
    if nose_var > 800:
        scores['angry'] += 20
    
    # SURPRISE: Wide eyes + open mouth + high overall variance
    if num_eyes >= 2 and eye_openness > 0.55:
        scores['surprise'] += 40
    
    if edge_density > 0.07:
        scores['surprise'] += 30
    
    if mouth_var > 1000:
        scores['surprise'] += 30
    
    if forehead_mean > 130:
        scores['surprise'] += 25
    
    if num_contours > 3:
        scores['surprise'] += 15
    
    # FEAR: Tense features + moderate eye opening + worried expression
    if num_eyes >= 2 and 0.45 < eye_openness < 0.65:
        scores['fear'] += 30
    
    if 800 < forehead_var < 1100:
        scores['fear'] += 25
    
    if 15 < asymmetry < 25:
        scores['fear'] += 25
    
    if gradient_magnitude > 22 and gradient_magnitude < 28:
        scores['fear'] += 25
    
    if eyes_var > 850:
        scores['fear'] += 20
    
    # DISGUST: Scrunched nose + specific facial tension + asymmetry
    if nose_var > 1000:
        scores['disgust'] += 40
    
    if asymmetry > 22:
        scores['disgust'] += 30
    
    if 0.04 < edge_density < 0.055:
        scores['disgust'] += 25
    
    if eyes_var > 950 and mouth_var < 700:
        scores['disgust'] += 20
    
    if smile_confidence == 0 and num_contours > 2:
        scores['disgust'] += 20
    
    # NEUTRAL: Balanced features + moderate values + relaxed appearance
    if smile_confidence == 0 and num_eyes >= 1:
        scores['neutral'] += 25
    
    if 600 < forehead_var < 950:
        scores['neutral'] += 20
    
    if 450 < mouth_var < 750:
        scores['neutral'] += 20
    
    if asymmetry < 12:
        scores['neutral'] += 25
    
    if 18 < gradient_magnitude < 24:
        scores['neutral'] += 20
    
    # Normalize scores to float
    total = sum(scores.values())
    if total > 0:
        emotions = {k: float((v/total)*100) for k, v in scores.items()}
    else:
        emotions = {k: float(100/7) for k in scores.keys()}
    
    dominant = max(emotions, key=emotions.get)
    
    return emotions, dominant

def hybrid_emotion_detector(frame):
    """
    Hybrid approach: Use DeepFace if available, fallback to advanced Haar
    """
    if DEEPFACE_AVAILABLE:
        return deepface_emotion_detector(frame)
    else:
        return advanced_haar_emotion_detector(frame)

def gen_frames():
    """Generate video frames with emotion overlay"""
    global session_stats
   
    while True:
        success, frame = camera.read()
        if not success:
            break
       
        session_stats['total_frames'] += 1
       
        try:
            emotions, dominant_emotion = hybrid_emotion_detector(frame)
            neutro_data = calculate_neutrosophic_values(emotions)
           
            emotion_buffer.append(dominant_emotion)
           
            if len(emotion_buffer) > 0:
                stable_emotion = max(set(emotion_buffer), key=emotion_buffer.count)
            else:
                stable_emotion = dominant_emotion
           
            session_stats['emotion_counts'][stable_emotion] = \
                session_stats['emotion_counts'].get(stable_emotion, 0) + 1
           
            # Store with native Python types
            neutro_history.append({
                'T': float(neutro_data[stable_emotion]['T']),
                'I': float(neutro_data[stable_emotion]['I']),
                'F': float(neutro_data[stable_emotion]['F']),
                'emotion': stable_emotion,
                'timestamp': float(time.time())
            })
           
            # Draw overlay
            overlay = frame.copy()
            h, w = frame.shape[:2]
           
            cv2.rectangle(overlay, (10, 10), (w-10, 120), (20, 20, 40), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
           
            cv2.putText(frame, f"Emotion: {stable_emotion.upper()}",
                       (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {emotions[stable_emotion]:.1f}%",
                       (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
           
            neutro = neutro_data[stable_emotion]
            cv2.putText(frame, f"T:{neutro['T']:.2f} I:{neutro['I']:.2f} F:{neutro['F']:.2f}",
                       (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 1)
           
        except Exception as e:
            cv2.putText(frame, "Analyzing...", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
       
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_data')
def emotion_data():
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'camera error'})
   
    try:
        emotions, dominant_emotion = hybrid_emotion_detector(frame)
        confidence = emotions[dominant_emotion]
        neutro_data = calculate_neutrosophic_values(emotions)
       
        emotion_buffer.append(dominant_emotion)
        stable_emotion = max(set(emotion_buffer), key=emotion_buffer.count) if emotion_buffer else dominant_emotion
       
        elapsed_time = time.time() - session_stats['start_time']
        fps = session_stats['total_frames'] / elapsed_time if elapsed_time > 0 else 0
       
        # Convert to native types before returning
        response_data = {
            'stable_emotion': stable_emotion,
            'dominant_emotion': dominant_emotion,
            'confidence': float(confidence),
            'emotions': convert_to_native_types(neutro_data),
            'session_stats': {
                'total_frames': int(session_stats['total_frames']),
                'emotion_counts': convert_to_native_types(session_stats['emotion_counts']),
                'elapsed_time': float(round(elapsed_time, 1)),
                'fps': float(round(fps, 1))
            }
        }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/emotion_history')
def emotion_history():
    # Convert all data to native Python types before returning
    history_list = [convert_to_native_types(item) for item in list(neutro_history)]
    return jsonify(history_list)

@app.route('/reset_session')
def reset_session():
    global session_stats
    session_stats = {
        'total_frames': 0,
        'emotion_counts': {},
        'start_time': time.time()
    }
    emotion_buffer.clear()
    neutro_history.clear()
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 NEUTROSOPHIC EMOTION RECOGNITION SYSTEM")
    print("="*60)
    print(f"Detection Method: {'DeepFace (Deep Learning)' if DEEPFACE_AVAILABLE else 'Advanced Haar Cascade'}")
    print(f"Camera Resolution: 640x480 @ 30 FPS")
    print(f"Server: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)