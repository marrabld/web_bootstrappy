from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import processing_form

import os

app = Flask(__name__)
app.secret_key = "satan secret key"

app.config['UPLOAD_FOLDER'] = '/tmp'
ALLOWED_EXTENSIONS = set(['csv', 'jpg'])


@app.route('/', methods=['GET', 'POST'])
def hello():
    # main_form = processing_form.main_form()
    # lbl_input_test  = ""
    return render_template('./index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_file', methods=['POST', 'GET'])
def upload_file():
    app.config['UPLOAD_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], request.form['txt_project_name'])
    print("file_upload")
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file parsed')
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print(file)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file',
            #                         filename=filename))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
