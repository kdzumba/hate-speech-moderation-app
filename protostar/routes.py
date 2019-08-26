import secrets
from PIL import Image
import os
from flask import render_template, url_for, request, redirect, flash, abort, jsonify
import pandas as pd
import protostar.features as features
from joblib import load
from flask_login import login_user, current_user, logout_user, login_required

from protostar.forms import RegistrationForm, LoginForm, UpdateAccountForm, PostForm
from protostar.database import User, Post
from protostar import app, db, bcrypt

model = load("protostar/model.joblib")


def predict(post):

    sen_feature = pd.DataFrame(features.get_sentiment_score(post), index=[0])
    dic_feature = pd.DataFrame(features.term_frequency(post), index=[0])
    tfidf_feature = features.get_tfidf_scores(post)

    post_df = sen_feature.merge(dic_feature, left_index=True, right_index=True)
    post_df = post_df.merge(dic_feature, left_index=True, right_index=True)
    post_df = post_df.merge(tfidf_feature, left_index=True, right_index=True)

    prediction = model.predict_proba(post_df)

    return prediction[0][1]


@app.route("/", methods=["POST", "GET"])
@login_required
def home():
    posts = Post.query.all()
    screened = False

    if request.method == "POST":

        content = request.form.get("post")
        action = request.form.get("action")

        if action == "detect":
            hate_level = round(predict(content), 4) * 100
            return render_template(
                "home.html", posts=posts, hate_level=hate_level, screened=True, post=content
            )

        elif action == "post":
            hate_level = round(predict(content), 4) * 100
            post = Post(content=content, author=current_user, hate_level=hate_level)
            db.session.add(post)
            db.session.commit()
            
            return redirect(url_for("home"))

    else:
        return render_template("home.html", posts=posts)


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


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    f_name, f_ext = os.path.splitext(form_picture.filename)
    picture_filename = random_hex + f_ext
    picture_path = os.path.join(
        app.root_path, "static/images/profile_pics", picture_filename
    )

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)
    return picture_filename


@app.route("/account", methods=["POST", "GET"])
@login_required
def account():
    form = UpdateAccountForm()

    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file

        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash("Account updated!", "success")
        return redirect(url_for("account"))
    elif request.method == "GET":
        form.username.data = current_user.username
        form.email.data = current_user.email

    image_file = url_for(
        "static", filename="images/profile_pics/" + current_user.image_file
    )
    return render_template(
        "account.html", title="Account", image_file=image_file, form=form
    )


@app.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template("post.html", post=post)


@app.route("/post/<int:post_id>/update")
@login_required
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = PostForm()
    return render_template("post_update.html", post=post, form=form)
