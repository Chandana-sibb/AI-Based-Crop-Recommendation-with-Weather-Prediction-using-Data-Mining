from django.shortcuts import render, redirect
from django.contrib.auth.models import User 
# Create your views here.
from django.contrib import messages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from . models import *
import pickle
from tensorflow.keras import layers, models 
import os
from dotenv import load_dotenv
import google.generativeai as genai


def index(request):

    return render(request,'index.html')

def about(request):
    
    return render(request,'about.html')

def login(request):
    if request.method=='POST':
        lemail=request.POST['email']
        lpassword=request.POST['password']

        d=Register.objects.filter(email=lemail,password=lpassword).exists()
        print(d)
        return redirect('userhome')
    else:
        return render(request,'login.html')

def registration(request):
    if request.method=='POST':
        Name = request.POST['Name']
        email=request.POST['email']
        password=request.POST['password']
        conpassword=request.POST['conpassword']
        

        print(Name,email,password,conpassword)
        if password==conpassword:
            rdata=Register(email=email,password=password)
            rdata.save()
            return render(request,'login.html')
        else:
            msg='Register failed!!'
            return render(request,'registration.html')

    return render(request,'registration.html')
    # return render(request,'registration.html')


def userhome(request):
    
    return render(request,'userhome.html')

def load(request):
   if request.method=="POST":
        file=request.FILES['file']
        global df
        df=pd.read_csv(file)
        messages.info(request,"Data Uploaded Successfully")
    
   return render(request,'load.html')

def view(request):
    # Assuming df is your DataFrame loaded from some source
    # df = pd.read_csv('your_file.csv')  # Example for loading CSV

    # Getting the first 100 rows
    dummy = df.head(100)

    # Extract columns and rows for rendering
    col = dummy.columns.tolist()  # Get column names as a list
    rows = dummy.values.tolist()  # Convert rows to a list of lists

    # Render the template with the data
    return render(request, 'view.html', {'col': col, 'rows': rows})
    
  
def preprocessing(request):

    global x_train,x_test,y_train,y_test,X,y
    if request.method == "POST":
        # size = request.POST['split']
        size = int(request.POST['split'])
        size = size / 100
        df.drop('date',axis=1,inplace=True)
        le = LabelEncoder()
        df['weather'] = le.fit_transform(df['weather'])

        #Preprocess Data for Machine Learning Development
        X = df.drop(['weather'], axis = 1)
        y = df['weather']

        oversample = SMOTE(random_state=1)
        X_final, Y_final = oversample.fit_resample(X, y)

        x_train, x_test, y_train, y_test = train_test_split(X_final, Y_final, test_size = 0.2, random_state = 10)
        x_train.shape, x_test.shape, y_train.shape, y_test.shape

        messages.info(request,"Data Preprocessed and It Splits Succesfully") 
    return render(request,'preprocessing.html')
 

def model1(request):
    if request.method == "POST":

        model1 = request.POST['algo']

        if model1 == "1":
            lr = LogisticRegression(max_iter=2000)
            lr.fit(x_train,y_train)
            y_pred = lr.predict(x_test)
            acc_lr=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of LogisticRegression :  ' + str(acc_lr)
            return render(request,'model1.html',{'msg':msg})

        elif model1 == "2":
            with open('CNN_model.pkl', 'rb') as fp:
                mod=pickle.load(fp)
            y_pred = mod.predict(x_test)
            y_pred_classes = y_pred.argmax(axis=-1)  # Get the index of the highest probability
            accuracy = accuracy_score(y_test, y_pred_classes)
            print(f"Accuracy: {accuracy}")
            msg = 'Accuracy of CNN :  ' + str(accuracy)
            return render(request,'model1.html',{'msg':msg})      
        
        elif model1 == "3":
            mlp=MLPClassifier(max_iter=2000)
            mlp.fit(x_train,y_train)
            y_pred = mlp.predict(x_test)
            acc_mlp=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of MLP :  ' + str(acc_mlp)
            return render(request,'model1.html',{'msg':msg}) 
        
        elif model1 == "4":
            rf_hyp = RandomForestClassifier(max_depth = None, max_features = 'log2', min_samples_leaf = 1, min_samples_split = 2, n_estimators = 200)
            rf_hyp.fit(x_train, y_train)
            y_pred = rf_hyp.predict(x_test)
            acc_rf=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of Random Forest :  ' + str(acc_rf)
            return render(request,'model1.html',{'msg':msg})  
        
        elif model1 == "5":
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            y_pred = dt.predict(x_test)
            acc_dt=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of Decision Tree :  ' + str(acc_dt)
            return render(request,'model1.html',{'msg':msg})   
    return render(request,'model1.html')

load_dotenv() 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

import google.generativeai as genai
from django.shortcuts import render
from sklearn.ensemble import RandomForestClassifier

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def prediction(request):
    if request.method == 'POST':
        # Get input features from the form
        f1 = float(request.POST['precipitation'])
        f2 = float(request.POST['temp_max'])
        f3 = float(request.POST['temp_min'])
        f4 = float(request.POST['wind'])

        # Prepare the prediction input
        PRED = [[f1, f2, f3, f4]]

        # Initialize and fit the Random Forest Classifier
        rf_hyp = RandomForestClassifier(max_depth=None, max_features='log2',
                                        min_samples_leaf=1, min_samples_split=2,
                                        n_estimators=200)
        rf_hyp.fit(x_train, y_train)
        RES = rf_hyp.predict(PRED)[0]

        # Map prediction result to weather condition
        weather_conditions = {0: 'Drizzle', 1: 'Fog', 2: 'Rain', 3: 'Snow', 4: 'Sun'}
        weather_type = weather_conditions.get(RES, 'Sun')
        weather_msg = f'The Weather is going to be: {weather_type}'

        # Prepare Gemini prompt for crop recommendation and farming process
        prompt = (f"Suggest the top 5 suitable crops for {weather_type} weather conditions. "
                  f"Provide detailed farming processes for each crop including planting, growing, harvesting, and care tips.")

        # Generate Gemini response
        response = model.generate_content(prompt)
        crop_recommendations = response.text

        context = {
            'weather_msg': weather_msg,
            'weather_type': weather_type,
            'crop_recommendations': crop_recommendations
        }

        return render(request, 'result.html', context)

    return render(request, 'prediction.html')
def result(request):
    return render(request, 'result.html')