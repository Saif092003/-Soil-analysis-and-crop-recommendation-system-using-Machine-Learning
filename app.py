from flask import Flask,render_template,request,redirect,url_for,session
import requests
import MySQLdb
from flask import send_file
from datetime import datetime
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from test_mobilenet import process_image, process


label={"apple":0,"Banana":1,"Blackgram":2,"Chickpea":3,"Coconut":4,"coffee":5,"Cotton":6,"Grapes":7,"Jute":8,"Kidney Beans":9,"Lentil":10,"Maize":11,"Mango":12,"Mouth Beans":13,"Mungbeans":14,"Muskmelon":15,"Orange":16,"Papaya":17,"Pigeon Peas":18,"Pomegranate":19,"Rice":20,"Watermelon":21}


UPLOAD_FOLDER = './static/Schemes/'
ALLOWED_EXTENSIONS = {'jpg','jpeg','png'}
mydb = MySQLdb.connect(host='localhost',user='root',passwd='root',db='croppred')
conn = mydb.cursor()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():

        if not session.get('logged_in'):
                return render_template("index.html")
                
        else:
                return render_template('userhomepage.html')
        
            

@app.route("/uregpage")
def uregpage():
        return render_template("reg.html")
@app.route("/uloginpage")
def uloginpage():
        return render_template("uloginpage.html")
def get_key1(val): 
    for key, value in label.items(): 
         if val == value: 
             return key 
@app.route("/predict",methods=['POST','GET'])
def predict():
        
        n=request.form['n']
        p=request.form['p']
        k=request.form['k']
        temperature=request.form['temperature']
        humidity=request.form['humidity']
        ph=request.form['ph']
        rainfall=request.form['rainfall']
        crop_data=pd.read_csv("Crop_recommendation.csv")
        crop_data.rename(columns = {'label':'Crop'}, inplace = True)
        crop_data = crop_data.dropna()
        x = crop_data[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
        target = crop_data['Crop']
        print("Target==",target)
        y = pd.get_dummies(target)
        print("Y==",y)
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state= 0)
        
        gb_clf = RandomForestClassifier(n_estimators = 100)
        MultiOutputClassifier(estimator=RandomForestClassifier(), n_jobs=-1)
        model = MultiOutputClassifier(gb_clf, n_jobs=-1)
        model.fit(x_train, y_train)
        x_test=[int(n),int(p),int(k),float(temperature),float(humidity),float(ph),float(rainfall)]
        x_test=np.array(x_test)
        x_test=x_test.reshape(1,-1)
        ypred=model.predict(x_test)
        print("Y-pred==",ypred)
        #label1=ypred.argmax(axis=1)
        #print("Predicted crop==",label)
        actallabel=get_key1(np.argmax(ypred))
        # fert=""
        # if actallabel=="Rice":
        #         fert="It is enough to apply 12.5 kg zinc sulphate /ha, if green manure (6.25 t/ha) or enriched FYM is applied. \n For saline and sodic acid 37.5 kg ZnSO4 /ha can be applied. \n Apply 500 kg of gypsum/ha (as source of Ca and S nutrients) at last ploughing. \n Application of 50 kg FeSO4 + 12.5 t FYM /ha, if the soil is deficient in Fe.For Cauvery delta zone, application of 5 kg CuSO4 /ha can be recommended."
        # if actallabel=="Maize":
        #         fert="Apply NPK fertilizers as per soil test recommendation as far as possible. If soil test recommendation is not available adopt a blanket recommendation of 135:62.5:50 NPK kg/ha, ZnSO4 at 37.5 kg/ha. \n Apply quarter of the dose of N; full dose of P2O and K2O basally before sowing. \n In the case of ridge planted crop, open a furrow 6 cm deep on the side of the ridge, at two thirds the distance from the top of the ridge. \n Apply the fertilizer mixture along the furrows evenly and cover to a depth of 4 cm with soil. \n If bed system of planting is followed, open furrows 6 cm deep at a distance of 60 cm apart. \n Place the fertilizer mixture along the furrows evenly and cover to a depth of 4 cm with soil."
        # if actallabel=="Chickpea":
        #         fert="15-20 kg N, 40 kg P2O5, 20 kg S, 1.0 kg Ammonium Molybdate and 5 tons FYM/ha. \n Spray of 2percent urea/DAP at flowering stage (70 DAS) and 10 days thereafter"
        # if actallabel=="Kidney Beans":
        #         fert="Apply Nitrogen@40kg/acre and Phosphorus@25kg/acre in form Urea@87kg and SSP@150kg/acre. \n  Do soil testing before sowing for accurate fertilizer application"
        # if actallabel=="Lentil":
        #         fert="The recommended dose of fertilizers is 20kg N, 40kg P, 20 kg K and 20kg S/ha. In soils low in Zinc, soil application of 20 kg ZnSO4 is recommended under rainfed and and late sown conditions. Foliar spray of 2 percent urea improves yield."
        # if actallabel=="Mouth Beans":
        #         fert="Yield levels of Moth bean have been observed to be increased by the applications of P2O5 up to 40 kg ha-1 at the sowing process. The applications of 10 kg N+40 kg P2O5 per hectare have proved the effective starter dose, hence, may be applied with."
        # if actallabel=="apple":
        #         fert="Urea is the most frequently used form of nitrogen for foliar application"
        # if actallabel=="Banana":
        #         fert="a balanced fertilizer with equal parts nitrogen, phosphorus and potassium is recommended"
        # if actallabel=="Blackgram":
        #         fert="Blackgram being a leguminous crop requires a dose of 20 Kg/ha Nitrogen, 40 Kg/ha Phosphorous, 20 Kg/ha Potash and 20 Kg/ha Sulphur to get full potential of high yielding varieties."
        # if actallabel=="Coconut":
        #         fert="From 5th year onwards, apply 50 kg of FYM or compost or green manure. 1.3 kg urea (560 g N), 2.0 kg super phosphate (320 g P2O5) and 2.0 kg muriate of potash (1200 g K2O) in two equal splits during June – July and December – January."
        # if actallabel=="coffee":
        #         fert="Coffee grounds contain several key nutrients needed by plants, including nitrogen, potassium, magnesium, calcium, and other trace minerals."
        # if actallabel=="Cotton":
        #         fert="normal fertilizer rates 50N: 30P: 35K per acre. (Urea 116 kg; SSP 188 kg and 60 kg MOP). Apply all SSP as soil basal dressing before planting ."
        # if actallabel=="Grapes":
        #         fert="Urea (46-0-0) at 2 to 3 ounces (1/2 cup) or bloodmeal (12-0-0) at 8 ounces (1 ½ cups) per vine will supply the desired amounts of nitrogen"
        # if actallabel=="Jute":
        #         fert="spray 8 kg of urea as 2 per cent urea solution (20 g urea in one litre of water) "
        # if actallabel=="Jute":
        #         fert="spray 8 kg of urea as 2 per cent urea solution (20 g urea in one litre of water) "
        # if actallabel=="Mango":
        #         fert="Apply 200:30:300 g N: P2O5: K2O / plant using water soluble fertilizers along with 25 litres of water/day."
        # if actallabel=="Mungbeans":
        #         fert="The fertilizer is composed of the following components in parts by weight: 100 to 105 parts of ammonium sulfate, 296 to 306 parts of urea, 117 to 127 parts of monoammonium phosphate, 95 to 105 parts of calcium superphosphate, 5 to 15 parts of calcium magnesium phosphor fertilizer"
        # if actallabel=="Muskmelon":
        #         fert="Apply 10 to 15 tonnes of farmyard manure, 50 kg of N (110 kg of Urea), 25 kg of P2)05 (155 kg of Single Superphosphate) and 25 kg of K20, (40 kg of Muriate of Potash) per acre to the directly seeded crop. The farmyard manure should be added 10-15 days before sowing."
        # if actallabel=="Orange":
        #         fert="For a recently planted orange tree in a pot, it's best to use a balanced, slow-release granular fertilizer with a ratio such as 10-10-10 or 14-14-14, which contains equal proportions of nitrogen, phosphorus, and potassium. This balanced fertilizer will support overall growth and root development."
        # if actallabel=="Papaya":
        #         fert="Apply 110 g of urea, 300 g of super phosphate and 80 g of muriate of potash to each plant per application."
        # if actallabel=="Pigeon Peas":
        #         fert="The recommended appropriate fertilizer application rate for a pigeon pea crop is: Rainfed: 25- 30Kg N + 40 Kg P + 30 kg K+ 10 kg S/ha. Irrigated : 25 Kg N + 50 Kg P+ 25 Kg K+ 20 Kg S/ha."
        # if actallabel=="Pomegranate":
        #         fert="Pomegranate, being an orchard crop, is a heavy feeder of nutrients. The recommended fertiliser dose is 600–700 gm of N, 200–250 gm of P2O5 and 200–250 gm of K2O per tree per year. In order to meet these nutritional needs, pomegranate growers should plan and follow the fertiliser management practices in a proper manner."
        # if actallabel=="Watermelon":
        #         fert="apply a fertilizer high in phosphorous, such as 10-10-10, at a rate of 4 pounds per 1,000 square feet (60 to 90 feet of row). Make a trench on the planting bed 4 to 6 inches deep and 2 inches from the side of the row."
        
        print("actallabel==",actallabel)
        return render_template("croppage1.html",msg=actallabel)

@app.route("/imagepage")
def imagepage():
        return render_template("imagepage.html")


@app.route("/predictimg",methods=['GET','POST'])
def predictimg():
        print("Entered")
        print("Entered here")
        if request.method == 'POST':
                uploaded_file = request.files['file']
                
                #file = request.files['file'] # fet input
                filename = uploaded_file.filename
                print("Filename==",filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                uploaded_file.save(file_path)
                img = process_image(uploaded_file.stream) 

                pred = process(img)
                return render_template("result.html", pred_output = pred,img_src=UPLOAD_FOLDER + uploaded_file.filename)
        

@app.route("/croppage1")
def croppage1():
        return render_template("croppage.html")
@app.route("/croppage")
def croppage():
        return render_template("croppage1.html")

@app.route("/ureg",methods=['POST'])
def ureg():
        name=request.form['name']
        uname=request.form['uname']
        passw=request.form['pass']
        
        mono=request.form['mono']
        email=request.form['email']
       
        age=request.form['age']
        gen=request.form['gen']
        
        cmd="SELECT * FROM farmer WHERE uname='"+uname+"' or email='"+email+"' "
        print(cmd)
        conn.execute(cmd)
        cursor=conn.fetchall()
        isRecordExist=0
        for row in cursor:
                isRecordExist=1
        if(isRecordExist==1):
                msg="Record Already Exist"
                return render_template("uregpage.html",data=msg)
        else:
                cmd="INSERT INTO farmer(name,uname,pass,mono,email,age,gen) Values('"+str(name)+"','"+str(uname)+"','"+str(passw)+"','"+str(mono)+"','"+str(email)+"','"+str(age)+"','"+str(gen)+"')"
                print(cmd)
                #print("Inserted Successfully")
                conn.execute(cmd)
                mydb.commit()
                msg="Added Successfully"
                print("msg==",msg)
                session['msg']=msg
                return redirect(url_for('index'))

@app.route("/ulogin",methods=['POST'])
def ulogin():
        uname=request.form['uname']
        passw=request.form['pass']
       
        

        
        cmd="SELECT * FROM farmer WHERE uname='"+uname+"' and pass='"+passw+"'"
        print(cmd)
        conn.execute(cmd)
        cursor=conn.fetchall()
        isRecordExist=0
        if len(cursor)>0:
                for row in cursor:
                        isRecordExist=1
            
        if(isRecordExist==1):
                session['uname']=uname
                session['utype']='user'
                session['logged_in']=True
                return redirect(url_for('index'))
        else:
                return render_template("uloginpage.html",msg="Incorret Password")


@app.route("/logout")
def log_out():
    session.clear()
    return redirect(url_for('index'))
    


                        
# start() 
if __name__=="__main__":
        app.run(host='0.0.0.0', port=5000,debug=True)
        

#flask run --host=0.0.0.0
