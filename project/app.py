from flask import Flask, render_template, request, redirect, flash, url_for, jsonify, session
from werkzeug.utils import secure_filename
import numpy as np
import sys
import os

sys.path.append('../..')
print(os.getcwd())
import lib.bootstrappy.libbootstrap.spectralmodel as spectralmodel
import lib.bootstrappy.libbootstrap.spectra_generator as spectra_generator

app = Flask(__name__)
app.secret_key = "satan's secret key"

app.config['UPLOAD_FOLDER'] = '/tmp'
ALLOWED_EXTENSIONS = {'csv'}


@app.before_request
def session_management():
    # make the session last indefinitely until it is cleared
    session.permanent = True


@app.route('/api/data/training', methods=['GET', 'POST'])
def data_training():
    return jsonify(session['DATA'])


@app.route('/api/data/processed', methods=['GET', 'POST'])
def data_processed():
    return jsonify(session['PROCESSED_DATA'])


@app.route('/', methods=['GET', 'POST'])
def index():
    if not 'DATA' in session:
        session['DATA'] = {'data': [0, 1, 2],
                           'label': [2, 5, 10]}
        session['PROCESSED_DATA'] = {'data': [0, 1, 2],
                                     'label': [2, 5, 10]}
    # main_form = processing_form.main_form()
    # print(session['DATA'])
    return render_template('./index.html',
                           data=session['DATA'],
                           processed_data=session['PROCESSED_DATA'])


def calc_bootstraps(filename, num_realizations=300):
    sm = spectralmodel.BuildSpectralModel(filename)
    sm.build()
    sg = spectra_generator.GenerateRealisation(sm, num_realizations)
    rrs = sg.gen_Rrs()

    np.savetxt('bootstrap.csv', np.real(rrs), delimiter=',')

    _tmp = np.loadtxt('bootstrap.csv', delimiter=',')
    x = np.real(_tmp[0, :])
    y = np.real(_tmp[1:, :])
    my_processed_data = {'data': y.tolist(),
                         'label': x.tolist()}
    session['PROCESSED_DATA'] = my_processed_data


def publish_data(filename):
    _tmp = np.loadtxt(filename, delimiter=',')
    x = _tmp[0, :]
    y = _tmp[1:, :]
    data = {'data': y.tolist(),
            'label': x.tolist()}
    session['DATA'] = data


def publish_processed_data(filename):
    # _tmp = np.loadtxt(filename, delimiter=',')
    _tmp = np.loadtxt('bootstrap.csv', delimiter=',')
    x = _tmp[0, :]
    y = _tmp[1:, :]
    processed_data = {'data': y.tolist(),
                      'label': x.tolist()}
    session['PROCESSED_DATA'] = processed_data


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
            # return redirect(url_for('index') + '#tab_processing')
            # return redirect(url_for('index'))
            # index()
            return render_template('./index.html',
                                   data=session['DATA'],
                                   processed_data=session['PROCESSED_DATA'])


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
