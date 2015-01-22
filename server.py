from flask import Flask
app = Flask(__name__)

from flask import render_template, url_for

# import assignment1 as a1

# data = a1.get_data()

@app.route('/')
def index():
  return render_template('index.html', data=[1,2,3])

if __name__ == "__main__":
    app.run()