import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from kneed import KneeLocator
from flask import *
import mysql.connector
db=mysql.connector.connect(host='localhost',user="root",password="",port='3306',database='crypto')
cur=db.cursor()


app=Flask(__name__)
app.secret_key = "fghhdfgdfgrthrttgdfsadfsaffgd"

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select count(*) from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        # cur.execute(sql)
        # data=cur.fetchall()
        # db.commit()
        x=pd.read_sql_query(sql,db)
        print(x)
        print('########################')
        count=x.values[0][0]

        if count==0:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            s="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            z=pd.read_sql_query(s,db)
            session['email']=useremail
            pno=str(z.values[0][4])
            print(pno)
            name=str(z.values[0][1])
            print(name)
            session['pno']=pno
            session['name']=name
            return render_template("userhome.html",myname=name)
    return render_template('login.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                msg="Registered successfully","success"
                return render_template("login.html",msg=msg)
            else:
                msg="Details are invalid","warning"
                return render_template("registration.html",msg=msg)
        else:
            msg="Password doesn't match", "warning"
            return render_template("registration.html",msg=msg)
    return render_template('registration.html')

@app.route('/load data',methods = ["POST","GET"])
def load_data():
    global df, dataset
    if request.method == "POST":
        data = request.files['file']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load data.html', msg=msg)
    return render_template('load data.html')

@app.route('/view data',methods = ["POST","GET"])
def view_data():
    df = pd.read_csv('final_data.csv')
    df.head(2)
    return render_template('view data.html',col_name = df.columns,row_val = list(df.values.tolist()))

@app.route('/model',methods = ['GET',"POST"])
def model():
    global x_train,x_test,y_train,y_test
    if request.method == "POST":
        model = int(request.form['selected'])
        print(model)
        df = pd.read_csv('final_data.csv')
        df.drop('Unnamed: 0',axis=1,inplace=True)
        print(df.columns)
        print('#######################################################')
        X = df.drop(['Pred'], axis =1)
        y = df.Pred
        x_train,x_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state  =101)
        print(df)
        if model == 1:
            from sklearn.tree import DecisionTreeClassifier 
            dt = DecisionTreeClassifier(random_state=12345)
            dt = dt.fit(x_train,y_train)
            y_pred = dt.predict(x_test)
            acc_dts=accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by DecisionTreeClassifier is ' + str(acc_dts) + str('%')
            return render_template('model.html',msg=msg)
        elif model ==2:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier()
            rf = rf.fit(x_train,y_train)
            y_pred = rf.predict(x_test)
            acc_rf=accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by RandomForestClassifier is ' + str(acc_rf) + str('%')
            return render_template('model.html',msg=msg)
        elif model ==3:
            from sklearn.ensemble import AdaBoostClassifier
            adb = AdaBoostClassifier()
            adb = adb.fit(x_train,y_train)
            y_pred = adb.predict(x_test)
            acc_adb=accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by AdaBoostClassifier  is ' + str(acc_adb) + str('%')
            return render_template('model.html',msg=msg)
        elif model ==4:
            from sklearn.neural_network import MLPClassifier
            mlp = MLPClassifier()
            mlp = mlp.fit(x_train,y_train)
            y_pred = mlp.predict(x_test)
            acc_mlp=accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by MLPClassifier  is ' + str(acc_mlp) + str('%')
            return render_template('model.html',msg=msg)
        elif model ==5:
            from sklearn.ensemble import ExtraTreesClassifier
            ets = ExtraTreesClassifier()
            ets = ets.fit(x_train,y_train)
            y_pred = ets.predict(x_test)
            acc_ets=accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by  ExtraTreesClassifier is ' + str(acc_ets) + str('%')
            return render_template('model.html',msg=msg)
    return render_template('model.html')

@app.route('/prediction' , methods=["POST","GET"])
def prediction():
    global x_train,y_train
    if request.method=="POST":
        f1=float(request.form['24h_volume_usd'])
        f2=float(request.form['available_supply'])
        f3=float(request.form['id'])
        f4=float(request.form['last_updated'])
        f5=float(request.form['market_cap_usd'])
        f6=float(request.form['max_supply'])
        f7=float(request.form['name'])
        f8=float(request.form['percent_change_1h'])
        f9=int(request.form['percent_change_24h'])
        f10=float(request.form['percent_change_7d'])
        f11=float(request.form['price_btc'])
        f12=float(request.form['price_usd'])
        f13=float(request.form['rank'])
        f14=float(request.form['symbol'])
        f15=float(request.form['total_supply'])
   

        lee=[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15]
        print(lee)

        import pickle
        from sklearn.tree import DecisionTreeClassifier
        model=DecisionTreeClassifier()
        global x_train,y_train
        model.fit(x_train,y_train)
        result=model.predict([lee])
        print(result)
        if result==0:
            msg="Financial Risk-Type == No-Risk Found"
        else:
            msg="Financial Risk-Type == Risk Found"
        return render_template('prediction.html', msg=msg)
    return render_template("prediction.html")

@app.route('/graph')
def graph ():

    # pic = pd.DataFrame({'Models':[]})
    # pic


    # plt.figure(figsize = (10,6))
    # sns.barplot(y = pic.Accuracy,x = pic.Models)
    # plt.xticks(rotation = 'vertical')
    # plt.show()

    return render_template('graph.html')


if __name__=="__main__":
    app.run(debug=True)