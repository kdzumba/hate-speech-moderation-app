from flask import render_template, url_for, request, redirect, flash, redirect, request
import pandas as pd
import protostar.features as features
from joblib import load
from flask_login import login_user, current_user, logout_user, login_required

from protostar.forms import RegistrationForm, LoginForm
from protostar.database import User, Post
from protostar import app, db, bcrypt

model = load("protostar/model.joblib")


@app.route("/register", methods=["POST", "GET"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    form = RegistrationForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode(
            "utf-8"
        )
        user = User(
            username=form.username.data, email=form.email.data, password=hashed_password
        )
        db.session.add(user)
        db.session.commit()
        flash(f"Your account has been created! {form.username.data}!", "success")
        return redirect(url_for("login"))
    return render_template("register.html", title="Register", form=form)


@app.route("/login", methods=["POST", "GET"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()

        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get("next")
            return redirect(next_page) if next_page else redirect(url_for("home"))
        else:
            flash("Login Unsuccessful. Please check username and password", "danger")
    return render_template("login.html", title="Login", form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("home"))


@app.route("/account")
@login_required
def account():
    return render_template("account.html", title="Account")


@app.route("/", methods=["POST", "GET"])
def home():

    if request.method == "POST":
        post = request.form.get("post")

        if post != None:
            sen_feature = pd.DataFrame(features.get_sentiment_score(post), index=[0])
            dic_feature = pd.DataFrame(features.term_frequency(post), index=[0])
            tfidf_feature = features.get_tfidf_scores(post)

            post_df = sen_feature.merge(dic_feature, left_index=True, right_index=True)
            post_df = post_df.merge(dic_feature, left_index=True, right_index=True)
            post_df = post_df.merge(tfidf_feature, left_index=True, right_index=True)

            prediction = model.predict_proba(post_df)

            return render_template("home.html", prediction=prediction[0][1], post=post)

    else:
        return render_template("home.html", prediction="")
