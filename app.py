from flask import Flask, Response, render_template, request
import numpy as np
import cv2

app = Flask(__name__)

# Precomputed transformation matrices
rgb_to_lms_matrix = np.array([[17.8824, 43.5161, 4.11935],
                              [3.45565, 27.1554, 3.86714],
                              [0.0299566, 0.184309, 1.46709]])

protanopia_matrix = np.array([[0, 2.02344, -2.52581],
                              [0, 1, 0],
                              [0, 0, 1]])

deuteranopia_matrix = np.array([[1.42319, -0.88995, 1.77557],
                                [0.67558, -0.42203, 2.82788],
                                [0.00267, -0.00504, 0.99914]])

tritanopia_matrix = np.array([[0.95451, -0.04719, 2.74872],
                              [-0.00447, 0.96543, 0.88835],
                              [-0.01251, 0.07312, -0.01161]])

protanopia_shift = np.array([[0, 0, 0],
                             [0.5, 1, 0],
                             [0.5, 0, 1]])

deuteranopia_shift = np.array([[1, 0.5, 0],
                               [0, 0, 0],
                               [0, 0.5, 1]])

tritanopia_shift = np.array([[1, 0, 0.7],
                             [0, 1, 0.7],
                             [0, 0, 0]])

lms_to_rgb_matrix = np.linalg.inv(rgb_to_lms_matrix)

# Precompute matrices
A1 = rgb_to_lms_matrix.T
A2_dict = {
    'protanopia': protanopia_matrix.T,
    'deuteranopia': deuteranopia_matrix.T,
    'tritanopia': tritanopia_matrix.T
}
A3 = lms_to_rgb_matrix.T
A4_dict = {
    'protanopia': protanopia_shift.T,
    'deuteranopia': deuteranopia_shift.T,
    'tritanopia': tritanopia_shift.T
}

camera = cv2.VideoCapture(0)
global color_mode
color_mode = None

def process_frame(frame, mode):
    lms_array = np.dot(frame, A1)
    simulated_rgb = np.dot(np.dot(lms_array, A2_dict[mode]), A3)
    compensated_rgb = np.clip(simulated_rgb + np.dot(frame - simulated_rgb, A4_dict[mode]), 0, 255).astype(np.uint8)
    return cv2.cvtColor(compensated_rgb, cv2.COLOR_RGB2BGR)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        if color_mode:
            processed_frame = process_frame(frame, color_mode)
        else:
            processed_frame = frame

        ret, buffer = cv2.imencode('.jpg', cv2.flip(processed_frame, 1))
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['POST', 'GET'])
def tasks():
    global color_mode
    if request.method == 'POST':
        if request.form.get('pro') == 'Protanopia':
            color_mode = 'protanopia'
        elif request.form.get('deu') == 'Deuteranopia':
            color_mode = 'deuteranopia'
        elif request.form.get('tri') == 'Tritanopia':
            color_mode = 'tritanopia'
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif request.method == 'GET':
        return render_template('index.html')

    return render_template('index.html')

if __name__ == '__main__':
    app.run()
