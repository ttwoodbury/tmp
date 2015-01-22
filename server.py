from flask import Flask
app = Flask(__name__)

from flask import render_template, url_for

import assignment1 as a1

print a1.df_stats.to_dict()

@app.route('/')
def index():
  return render_template('index.html', data=a1.df_stats.to_dict())

if __name__ == "__main__":
    app.run()