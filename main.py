
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    df_dt = pd.read_csv('Dataset/MT21149_test_result_IR_DT.csv')
    datatest = pd.read_csv('Dataset/test.csv')

    int_features = [x for x in request.form.values()]


    #print(int_features)
    str_ans = ''
    upload = "By Decision Tree Classifier"
    ans = "FALSE"
    for k in int_features:
        for i, j in zip(datatest['text'], df_dt['target']):
             if (k == i):
                 if(j==0):
                    ans="FALSE"

                 else:
                    ans = "TRUE"
        str_ans += ans + ' '

    str_ans = str_ans.split(" ")

    for i in range(len(str_ans)-1):
        upload +=  '\n'+' Tweet '+ str(i+1)+ " is " + str_ans[i] + ' , '

    upload_knn="By KNN Classifier"
    df_knn = pd.read_csv('Dataset/MT21149_test_result_IR_knn.csv')
    ans = "FALSE"
    str_ans=""
    for k in int_features:
        for i, j in zip(datatest['text'], df_knn['target']):
            if (k == i):
                if (j == 0):
                    ans = "FALSE"

                else:
                    ans = "TRUE"
        str_ans += ans + ' '

    str_ans = str_ans.split(" ")

    for i in range(len(str_ans) - 1):
        upload_knn +=  '\n'+' Tweet ' + str(i + 1) + " is " + str_ans[i] + ' , '


    upload_tfidf="By Tf-Idf Classifier"
    df_tfidf = pd.read_csv('Dataset/MT21149_test_result_IR_tfidf.csv')
    ans = "FALSE"
    str_ans=""
    for k in int_features:
        for i, j in zip(datatest['text'], df_tfidf['target']):
            if (k == i):
                if (j == 0):
                    ans = "FALSE"

                else:
                    ans = "TRUE"
        str_ans += ans + ' '

    str_ans = str_ans.split(" ")

    for i in range(len(str_ans) - 1):
        upload_tfidf +=  '\n'+' Tweet ' + str(i + 1) + " is " + str_ans[i] + ' , '


    upload_naive_bayes="By Naive Bayes Classifier"
    df_tfidf = pd.read_csv('Dataset/MT21149_test_result_IR_naive_bayes.csv')
    ans = "FALSE"
    str_ans=""
    for k in int_features:
        for i, j in zip(datatest['text'], df_tfidf['target']):
            if (k == i):
                if (j == 0):
                    ans = "FALSE"

                else:
                    ans = "TRUE"
        str_ans += ans + ' '

    str_ans = str_ans.split(" ")

    for i in range(len(str_ans) - 1):
        upload_naive_bayes +=  '\n'+' Tweet ' + str(i + 1) + " is " + str_ans[i] + ' , '


    return render_template('index.html', prediction_text=upload,upload_knn=upload_knn,upload_tfidf=upload_tfidf,upload_naive_bayes=upload_naive_bayes)



if __name__ == "__main__":
    app.run(debug=True)
