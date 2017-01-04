from flask import Flask, render_template, request, redirect, flash, url_for, jsonify, session
from werkzeug.utils import secure_filename
from werkzeug.contrib.cache import SimpleCache
import numpy as np
import sys
import os

# cache = MemcachedCache(['127.0.0.1:112111'])
cache = SimpleCache(default_timeout=0)
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


@app.route('/api/data/training', methods=['GET'])
def data_training():
    try:
        my_dict = cache.get('DATA')
    except:
        my_dict = {'data': [], 'label': []}

    if not my_dict:
        my_dict = {'data': [], 'label': []}

    return jsonify(my_dict)


@app.route('/api/data/training/<int:i_iter>', methods=['GET'])
def data_training_array(i_iter):
    try:
        my_data = cache.get('DATA')['data'][i_iter]
        my_label = cache.get('DATA')['label']

        my_dict = {'data': my_data, 'label': my_label}
    except:
        my_dict = {'data': [], 'label': []}

    return jsonify(my_dict)


@app.route('/api/data/processed', methods=['GET'])
def data_processed():
    try:
        my_dict = cache.get('PROCESSED_DATA')
    except:
        my_dict = {'data': [], 'label': []}

    if not my_dict:
        my_dict = {'data': [], 'label': []}

    return jsonify(my_dict)


@app.route('/api/data/processed/<int:i_iter>', methods=['GET'])
def data_processed_array(i_iter):
    try:
        my_data = cache.get('PROCESSED_DATA')['data'][i_iter]
        my_label = cache.get('PROCESSED_DATA')['label']

        my_dict = {'data': my_data, 'label': my_label}

    except:
        my_dict = {'data': [], 'label': []}

    return jsonify(my_dict)


@app.route('/', methods=['GET', 'POST'])
def index():
    if not cache.get('DATA'):
        cache.set('DATA', {'data': [0, 1, 2],
                           'label': [2, 5, 10]})

        cache.set('PROCESSED_DATA', {'data': [0, 1, 2],
                                     'label': [2, 5, 10]})

    return render_template('./index.html',
                           data=cache.get('DATA'),
                           processed_data=cache.get('PROCESSED_DATA'))


def calc_bootstraps(filename, num_realizations=100):
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
    # session['PROCESSED_DATA'] = my_processed_data
    cache.set('PROCESSED_DATA', my_processed_data)


def publish_data(filename):
    _tmp = np.loadtxt(filename, delimiter=',')
    x = _tmp[0, :]
    y = _tmp[1:, :]
    data = {'data': y.tolist(),
            'label': x.tolist()}
    cache.set('DATA', data)


def publish_processed_data(filename):
    # _tmp = np.loadtxt(filename, delimiter=',')
    _tmp = np.loadtxt('bootstrap.csv', delimiter=',')
    x = _tmp[0, :]
    y = _tmp[1:, :]
    processed_data = {'data': y.tolist(),
                      'label': x.tolist()}
    cache.set('PROCESSED_DATA', processed_data)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_file', methods=['POST', 'GET'])
def upload_file():
    app.config['UPLOAD_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], request.form['txt_project_name'])
    print("file_upload")
    num_iters = np.int(request.form['txt_num_iters'])
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
            calc_bootstraps(os.path.join(app.config['UPLOAD_FOLDER'], filename), num_iters)
            publish_processed_data(app.config['UPLOAD_FOLDER'])
            # return redirect(url_for('index') + '#tab_processing')
            # return redirect(url_for('index'))
            # index()
            return render_template('./index.html',
                                   data=cache.get('DATA'),
                                   processed_data=cache.get('PROCESSED_DATA'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
