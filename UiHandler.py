from flask import Flask, render_template, url_for, request, session, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
from flask_jsglue import JSGlue
from ImageCartoonifier import Cartoonifier
import webbrowser

# configuring flask app
app = Flask(__name__)
jsGlue = JSGlue(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static\inputImages')
STYLE_IMAGE_FOLDER = os.path.join(APP_ROOT, 'static\styles\images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STYLE_IMAGE_FOLDER'] = STYLE_IMAGE_FOLDER
ALLOWED_EXTENSIONS = {'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


@app.route('/upload_file', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'content_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        if 'style_files' not in request.files:
            flash('No file part')
            return redirect(request.url)

        content_file = request.files['content_file']
        style_files = request.files.getlist("style_files")

        # if user does not select file, browser also
        # submit an empty part without filename
        if content_file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        check_style_file_names = True
        for style_file in style_files:
            if style_file and allowed_file(style_file.filename) and not style_file.filename == '':
                filename = secure_filename(style_file.filename)
                style_file.save(os.path.join(app.config['STYLE_IMAGE_FOLDER'], filename))
        if content_file and allowed_file(content_file.filename):
            filename = secure_filename(content_file.filename)
            content_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # res = request.form.getlist('default_style')

        content_path = app.config['UPLOAD_FOLDER'] + "\\" + content_file.filename
        style_paths = []
        for image in request.form.getlist('default_style'):
            style_paths.append(app.config['STYLE_IMAGE_FOLDER'] + "\\" + image + ".jpeg")
        for style_file in style_files:
            if style_file.filename != '':
                style_paths.append(app.config['STYLE_IMAGE_FOLDER'] + "\\" + style_file.filename)

        if len(content_path) == 0 or len(style_paths) == 0:
            return flash("please enter required files")

        cartoonifier = Cartoonifier(content_path, style_paths)
        cartoonifier.main()

    return send_from_directory(app.config['STYLE_IMAGE_FOLDER'], "stylized-image.png")


if __name__ == '__main__':
    app.secret_key = 'mysecret'
    app.run(debug=True, port=8000)


