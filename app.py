from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

# Sample users for login
users = {
    'user': {'username': 'user1', 'password': 'password123'},
    'admin': {'username': 'admin', 'password': 'adminpass'}
}

# Configuration and initial setup
nimgs = 10
imgBackground = cv2.imread("background.png")
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# In-memory complaints storage
complaints = []

# Helper functions
def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    predicted = model.predict(facearray)
    if len(predicted) == 0:
        return "Not Detected"
    return predicted[0]

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    return df['Name'], df['Roll'], df['Time'], len(df)

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names, rolls = [], []
    for user in userlist:
        name, roll = user.split('_')
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, len(userlist)

def get_greeting():
    # Get the current hour
    current_hour = datetime.now().hour
    # Determine greeting based on the current time
    if 5 <= current_hour < 12:
        return "Good Morning"
    elif 12 <= current_hour < 17:
        return "Good Afternoon"
    elif 17 <= current_hour < 21:
        return "Good Evening"
    else:
        return "Good Night"

# Flask Routes

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login_post', methods=['POST'])
def login_post():
    user_type = request.form['user_type']
    username = request.form['username']
    password = request.form['password']

    # Check user credentials
    if user_type in users and users[user_type]['username'] == username and users[user_type]['password'] == password:
        greeting = get_greeting()  # Get the greeting message based on the time
        if user_type == 'user':
            return render_template('user.html', greeting=greeting)
        elif user_type == 'admin':
            return render_template('admin.html', greeting=greeting)
    else:
        return "Invalid login credentials", 401

@app.route('/user_dashboard')
def user_dashboard():
    greeting = get_greeting()  # Get the greeting message based on the time
    return render_template('user.html', greeting=greeting)

@app.route('/admin_dashboard')
def admin_dashboard():
    greeting = get_greeting()  # Get the greeting message based on the time
    return render_template('admin.html', greeting=greeting)

@app.route('/student_detail')
def student_detail():
    userlist, names, rolls, l = getallusers()
    return render_template('home.html', names=names, rolls=rolls, l=l)

@app.route('/attendance')
def attendance():
    names, rolls, times, l = extract_attendance()
    return render_template('list.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', mess='No trained model found. Please add a new face to continue.')
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret and len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            if identified_person != "Not Detected":
                add_attendance(identified_person)
        
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while i < nimgs:
        ret, frame = cap.read()
        if ret:
            faces = extract_faces(frame)
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                cv2.imwrite(f"{userimagefolder}/{newusername}_{newuserid}_{i}.jpg", frame[y:y+h, x:x+w])
                i += 1
        j += 1
        if j % 30 == 0:
            print(f"{i}/{nimgs} images captured.")
    cap.release()
    train_model()
    return render_template('home.html', mess=f'Face of {newusername} added successfully.')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/submit_complain', methods=['POST'])
def submit_complain():
    name = request.form['name']
    email = request.form['email']
    complain_message = request.form['complain']
    complaint_id = len(complaints) + 1
    complaints.append({
        "id": complaint_id,
        "name": name,
        "email": email,
        "message": complain_message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    return '''
    <script>
        alert("Thank you! We will get back to you soon.");
        window.location.href = "/help";
    </script>
    '''

@app.route('/complain')
def complain():
    from datetime import datetime
    year = datetime.now().year
    return render_template('complain.html', complaints=complaints, year=year)

@app.route('/delete_complain/<int:complaint_id>', methods=['POST'])
def delete_complain(complaint_id):
    global complaints
    complaints = [complaint for complaint in complaints if complaint['id'] != complaint_id]
    return redirect(url_for('complain'))

@app.route('/download_attendance')
def download_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    excel_path = f'Attendance/Attendance-{datetoday}.xlsx'
    df.to_excel(excel_path, index=False)
    return send_file(excel_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
