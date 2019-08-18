from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap
import pandas as pd
import features
from joblib import load


app = Flask(__name__)
Bootstrap(app)

model = load("model.joblib")


@app.route("/", methods=["POST", "GET"])
def predict():

    if request.method == "POST":
        post = request.form.get("post")

        sen_feature = pd.DataFrame(features.get_sentiment_score(tweet), index=[0])
        dic_feature = pd.DataFrame(features.term_frequency(tweet), index=[0])
        tfidf_feature = features.get_tfidf_scores(tweet)

        post_df = sen_feature.merge(dic_feature, left_index=True, right_index=True)
        post_df = tweet_df.merge(dic_feature, left_index=True, right_index=True)
        post_df = tweet_df.merge(tfidf_feature, left_index=True, right_index=True)

        prediction = model.predict_proba(post_df)

        return render_template("index.html", prediction = prediction[0][1])

    else:
        return render_template("index.html", prediction="")


if __name__ == "__main__":
    app.run(debug=True)
