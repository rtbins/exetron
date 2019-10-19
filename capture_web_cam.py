import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import cv2
import requests
import numpy as np
from PIL import Image, ImageDraw

app = Flask(__name__)

app.config['LIVE_VIDEO'] = 'liveData/'
test_url = "http://localhost:5000/predict"

@app.route('/live_frame_capture')
def liveFrameCaptureFromVideo():
    vidcap = cv2.VideoCapture(0)
    count = 0
    config_frame_number = 3
    reading_frame = 0
    success, frame = vidcap.read()
    while success:
        # cv2.imshow('Our Live Sketcher', frame)
        # if cv2.waitKey(1) == 13: #13 is the Enter Key
        #     break
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
        # save frame as JPEG file
        cv2.imwrite(os.path.join(app.config['LIVE_VIDEO'], "frame%d.jpg" % count), frame)
        success, frame = vidcap.read()
        # cv2.imwrite('frame%d.jpg' % count, frame)  # save frame as JPEG file
        count += 1
        if (count % 3 == 0 ):
            content_type = 'image/jpeg'
            headers = {'content-type': content_type}    
            _, img_encoded = cv2.imencode('.jpg', frame)
            data = {}
            data["image"] = img_encoded.tostring()
            # TODO
            #data["name"] = "frame%d.jpg" % count
            requests.post(test_url, data=img_encoded.tostring(), headers=headers)

            # print frame
        #     showImage("LiveData/frame%d.jpg" % count, count)
        
    # Release camera and close windows
    vidcap.release()
    return redirect(url_for('/'))



if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5001)