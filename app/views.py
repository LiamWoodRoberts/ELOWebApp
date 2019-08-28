# Module Imports
from app import app,predictor

# Package Imports
from flask import render_template

# APP ROUTES
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/demo")
def demo():
    sample_cols,sample_vals,metric_cols,metric_vals,preds,percentile = predictor.get_demo_values()
    return render_template("demo.html",
                            sample_cols=sample_cols,
                            sample_vals=sample_vals,
                            metric_cols=metric_cols,
                            metric_vals=metric_vals,
                            pred=preds,
                            percentile=percentile)

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/nav")
def nav():
    return render_template("nav_bar.html")
