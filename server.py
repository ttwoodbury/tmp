from flask import Flask
app = Flask(__name__)

from flask import render_template, url_for

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run()