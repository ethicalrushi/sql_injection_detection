from wtforms import Form, FloatField,TextField, validators


class InputForm(Form):
    a = TextField(
        label='Input Query', default=1,
        validators=[validators.InputRequired()])
