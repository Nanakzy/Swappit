import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
from face_swap import face_swap

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Ensure both images are uploaded
    if 'original_image' not in request.files or 'new_face_image' not in request.files:
        return redirect(request.url)

    original_image_file = request.files['original_image']
    new_face_image_file = request.files['new_face_image']

    # Validate the files
    if original_image_file and allowed_file(original_image_file.filename) and new_face_image_file and allowed_file(new_face_image_file.filename):
        # Save the images
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_image.jpg')
        new_face_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'new_face_image.jpg')
        
        original_image_file.save(original_image_path)
        new_face_image_file.save(new_face_image_path)
        
        # Perform face swap
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_swapped.jpg')
        if face_swap(original_image_path, new_face_image_path, result_image_path):
            return redirect(url_for('display_result', filename='face_swapped.jpg'))

    return redirect('/')

@app.route('/display/<filename>')
def display_result(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
