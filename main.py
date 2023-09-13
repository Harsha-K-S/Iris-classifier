from flask import Flask, render_template, request, redirect, url_for
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('userInput.html')

@app.route('/results/<string:cls>')
def results(cls):
    return render_template('results.html', cls=cls)

@app.route('/submit', methods=['POST'])
def submit():
    cls = ""
    with open('iris_predict_model.pkl', 'rb') as file:
        clsf = pickle.load(file)
    if request.method == 'POST':
        seplen = float(request.form['sepal-len'])
        sepwid = float(request.form['sepal-wid'])
        petallen = float(request.form['petal-len'])
        petalwid = float(request.form['petal-wid'])
        pre = clsf.predict([[seplen, sepwid, petallen, petalwid]])
        if pre == 1:
            cls = "Versicolor"
        elif pre == 2:
            cls = "virginica"
        else:
            cls = "setosa"
    return redirect(url_for('results', cls=cls))

if __name__ == '__main__':
    app.run()


