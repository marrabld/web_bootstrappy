from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import numpy as np
import sys
import os

sys.path.append('../..')
print(os.getcwd())
import lib.bootstrappy.libbootstrap.spectralmodel as spectralmodel
import lib.bootstrappy.libbootstrap.spectra_generator as spectra_generator

app = Flask(__name__)
app.secret_key = "satan secret key"

app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['DATA'] = [1, 3, 4, 3, 5, 7]
app.config['PROCESSED_DATA'] = [1, 2, 3, 4, 5, 6]
ALLOWED_EXTENSIONS = {'csv'}


@app.route('/', methods=['GET', 'POST'])
def index():
    # main_form = processing_form.main_form()
    print(app.config['DATA'])
    return render_template('./index.html',
                           data=app.config['DATA'],
                           processed_data=app.config['PROCESSED_DATA'])


def calc_bootstraps(filename, num_realizations=300):
    sm = spectralmodel.BuildSpectralModel(filename)
    sm.build()
    sg = spectra_generator.GenerateRealisation(sm, num_realizations)
    rrs = sg.gen_Rrs()
    print('def calc_bootstrap')
    np.savetxt('bootstrap.csv', np.real(rrs), delimiter=',')

    _tmp = np.loadtxt('bootstrap.csv', delimiter=',')
    x = np.real(_tmp[0, :])
    y = np.real(_tmp[1:, :])
    my_processed_data = {'data': y.tolist(),
                         'label': x.tolist()}
    app.config['PROCESSED_DATA'] = my_processed_data


def publish_data(filename):
    _tmp = np.loadtxt(filename, delimiter=',')
    x = _tmp[0, :]
    y = _tmp[1:, :]
    data = {'data': y.tolist(),
            'label': x.tolist()}
    app.config['DATA'] = data


def publish_processed_data(filename):
    # _tmp = np.loadtxt(filename, delimiter=',')
    _tmp = np.loadtxt('bootstrap.csv', delimiter=',')
    x = _tmp[0, :]
    y = _tmp[1:, :]
    processed_data = {'data': y.tolist(),
            'label': x.tolist()}
    app.config['PROCESSED_DATA'] = processed_data


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
            # return
            publish_data(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            calc_bootstraps(os.path.join(app.config['UPLOAD_FOLDER'], filename), 200)
            publish_processed_data(app.config['UPLOAD_FOLDER'])
            return redirect(url_for('index') + '#tab_processing')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
