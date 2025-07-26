from flask import Flask, render_template, request, Response
from detect_image import detect_objects_in_image
from detect_video import detect_objects_in_video
from detect_camera import detect_from_camera
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['image']
    if file:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        result_path = detect_objects_in_image(path)
        return render_template('image_result.html', result_img=result_path)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        result_path = detect_objects_in_video(path)
        return render_template('video_result.html', result_video=result_path)

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_from_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
