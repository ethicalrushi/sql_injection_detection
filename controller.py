from form import InputForm
from flask import Flask, render_template, request
from compute import compute

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        result = compute(form.a.data)
                         
    else:
        result = None

    return render_template('index.html', form=form, result=result)

if __name__ == '__main__':
    app.run(debug = True)