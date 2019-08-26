from flask import Flask, render_template, request
from werkzeug import secure_filename
import os ,cv2
import collections as cl
import json

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])
UPLOAD_FOLDER = './img'
image_dir="img"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['POST'])
def test():
    if request.method == 'POST':
        img_file=request.files["post_data"]
        if img_file:
            filename = secure_filename(img_file.filename)
            print(filename)
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path="./img/"+os.listdir(image_dir)[0]
            #faces_list=Main(image_path)
            data=cl.OrderedDict()
            print("{}".format(json.dumps(data,indent=4)))
        return json.dumps(data,indent=4)


    #return render_template("index.html",s=10)


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=8080)
