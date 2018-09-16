from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired


class InputForm(FlaskForm):
    text_area = TextAreaField(validators=[DataRequired()])
    submit1 = SubmitField("Submit")


class FileForm(FlaskForm):
    txt_file = FileField('txt file',
             validators=[FileRequired(), FileAllowed(['txt'], 'txt only!')])
    submit2 = SubmitField('Upload')
