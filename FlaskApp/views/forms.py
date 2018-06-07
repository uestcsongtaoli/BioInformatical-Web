from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField


class FileForm(FlaskForm):
    txt_file = FileField('txt file', validators=[
        FileRequired(),
        FileAllowed(['txt'], 'txt only!')
    ])
    submit = submit = SubmitField('Upload')
