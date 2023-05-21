from flask import Flask,render_template,request
from mlmodel import *
import numpy as np 

app=Flask(__name__)
@app.route("/")

def home():
    return render_template("page A.html")

@app.route("/getprediction",methods=["POST"])
def getpredict():
    rnd=request.form['rnd']
    ad=request.form['ad']
    ms=request.form['ms']
    newob=[[rnd,ad,ms]]
    newobs=np.array(newob,dtype=int)
    print(newobs)
    model=makeprediction()
    yp=model.predict(newobs)[0]
    return render_template("page B.html",data=yp)
    
if(__name__=="__main__"):
    app.run(debug=True)
