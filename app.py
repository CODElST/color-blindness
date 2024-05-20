from flask import Flask, Response, render_template, request, redirect
import numpy as np
import cv2


app = Flask(__name__)

# RGB to LMS transformation matrix (equation M4)
rgb_to_lms_matrix = np.array([[17.8824, 43.5161, 4.11935],
                [3.45565, 27.1554, 3.86714],
                [0.0299566, 0.184309, 1.46709]])

# Color blindness simulation matrices
protanopia_matrix = np.array([[0, 2.02344, -2.52581],
                [0, 1, 0],
                [0, 0, 1]])

deuteranopia_matrix = np.array([[ 1.42319, -0.88995, 1.77557],
                [ 0.67558, -0.42203, 2.82788],
                [0.00267, -0.00504, 0.99914]])

tritanopia_matrix = np.array([[0.95451, -0.04719, 2.74872],
                [-0.00447, 0.96543, 0.88835],
                [-0.01251, 0.07312, -0.01161]])

# Create a video capture object for webcam
camera = cv2.VideoCapture(0)


global prota, deutero, trita
prota = False
deutero = False
trita = False

def gen_frames():  # generate frame by frame from camera
    while True:
        success, frame = camera.read() 
        original_array = np.array(frame)

        # Apply RGB to LMS transformation
        lms_array = np.dot(original_array, rgb_to_lms_matrix.T)

        simulated_lms_protanopia = np.dot(lms_array, protanopia_matrix.T)
        simulated_lms_deuteranopia = np.dot(lms_array, deuteranopia_matrix.T)
        simulated_lms_tritanopia = np.dot(lms_array, tritanopia_matrix.T)

        # Inverse transformation to RGB (equation M8)
        lms_to_rgb_matrix = np.linalg.inv(rgb_to_lms_matrix)
        simulated_rgb_protanopia = np.dot(simulated_lms_protanopia, lms_to_rgb_matrix.T)
        simulated_rgb_deuteranopia = np.dot(simulated_lms_deuteranopia, lms_to_rgb_matrix.T)
        simulated_rgb_tritanopia = np.dot(simulated_lms_tritanopia, lms_to_rgb_matrix.T)

        difference_protanopia = original_array - simulated_rgb_protanopia
        difference_deuteranopia = original_array - simulated_rgb_deuteranopia
        difference_tritanopia = original_array - simulated_rgb_tritanopia

        protanopia_shift = np.array([[0, 0, 0],
                                  [0.5, 1, 0],
                                  [0.5, 0, 1]])
        deuteranopia_shift = np.array([[1, 0.5, 0],
                                      [0, 0, 0],
                                      [0, 0.5, 1]])
        tritanopia_shift = np.array([[1, 0, 0.7],
                                    [0, 1, 0.7],
                                    [0, 0, 0]])
        
        shifted_protanopia = np.dot(difference_protanopia, protanopia_shift.T)
        shifted_deuteranopia = np.dot(difference_deuteranopia, deuteranopia_shift.T)
        shifted_tritanopia = np.dot(difference_tritanopia, tritanopia_shift.T)

        compensated_rgb_protanopia = simulated_rgb_protanopia + shifted_protanopia
        compensated_rgb_deuteranopia = simulated_rgb_deuteranopia + shifted_deuteranopia
        compensated_rgb_tritanopia = simulated_rgb_tritanopia + shifted_tritanopia

        compensated_rgb_protanopia = np.clip(compensated_rgb_protanopia, 0, 255).astype(np.uint8)
        compensated_rgb_deuteranopia = np.clip(compensated_rgb_deuteranopia, 0, 255).astype(np.uint8)
        compensated_rgb_tritanopia = np.clip(compensated_rgb_tritanopia, 0, 255).astype(np.uint8)

        # convert the array into a video frame
        compensated_rgb_protanopia = cv2.cvtColor(compensated_rgb_protanopia, cv2.COLOR_RGB2BGR)
        compensated_rgb_deuteranopia = cv2.cvtColor(compensated_rgb_deuteranopia, cv2.COLOR_RGB2BGR)
        compensated_rgb_tritanopia = cv2.cvtColor(compensated_rgb_tritanopia, cv2.COLOR_RGB2BGR)


        if success:     
            try:
                if prota:
                  ret, buffer = cv2.imencode('.jpg', cv2.flip(compensated_rgb_protanopia,1))
                elif deutero:
                  ret, buffer = cv2.imencode('.jpg', cv2.flip(compensated_rgb_deuteranopia,1))
                elif trita:
                  ret, buffer = cv2.imencode('.jpg', cv2.flip(compensated_rgb_tritanopia,1))
                  
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/',methods=['POST','GET'])
def tasks():
    global prota, deutero, trita
    if request.method == 'POST':
        if request.form.get('pro') == 'Protanopia':  
            prota = True
            deutero = False
            trita = False          
            return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        elif request.form.get('deu') == 'Deuteranopia':
            prota = False
            deutero = True
            trita = False
            return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        elif request.form.get('tri') == 'Tritanopia':
            prota = False
            deutero = False
            trita = True
            return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')                  
                 
    elif request.method=='GET':
        return render_template('index.html')

    return render_template('index.html')

if __name__ == '__main__':
    app.run()

