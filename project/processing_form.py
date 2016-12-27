from flask_wtf import Form
from wtforms import StringField
from wtforms.validators import DataRequired


class main_form(Form):
    txt_project_name = StringField('txt_project_name')