import subprocess as sb
import os
import sys
import pyttsx3 as s2t
import speech_recognition as sr
import datetime as dt
import wikipedia as wiki
import webbrowser as wb
import seaborn
import pandas as pd
import sklearn.metrics as result
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer, QTime, QDate, Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
from PheonixUI import Ui_Pheonix

engine = s2t.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()
def wishme():
    hour = int(dt.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")
    elif hour>=12 and hour<16:
        speak("Good afternoon!")
    else:
        speak("Good evening!")
    
    speak("How can i help u")

class Mainthread(QThread):
    def __init__(self):
        super(Mainthread,self).__init__()
    def run(self):
        self.taskexecution()

    def takeCommand(self):
    

        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listning....")
            r.pause_threshold = 0.5
            r.energy_threshold = 100
            audio = r.listen(source)

        try:
            print("Recognising...")
            self.query = r.recognize_google(audio, language='en-in')
            print(f"Input: {self.query}\n")

        except Exception as e:
            # print(e)
            print("Couldn't hear, Say that again after pressing start...")
            speak("Couldn't hear, Say that again after pressing start...")
            return"None"
        return self.query

    def taskexecution(self):
        def speak(audio):
            engine.say(audio)
            engine.runAndWait()
        def wishme():
            hour = int(dt.datetime.now().hour)
            if hour>=0 and hour<12:
                speak("Good Morning!")
            elif hour>=12 and hour<16:
                speak("Good afternoon!")
            else:
                speak("Good evening!")
            
            speak("I am Pheonix, How can i help u")
        if __name__ == "__main__":
            wishme()
            # while True:
            if 1:
                self.query = self.takeCommand().lower()
                
                if 'wikipedia' in self.query:
                    speak('searching wikipedia...')
                    self.query = self.query.replace("wikipedia", "")
                    results = wiki.summary(self.query, sentences=1)
                    speak("According to wikipedia....")
                    print(results)
                    speak(results)
                elif 'open youtube' in self.query:
                    wb.open("youtube.com")
                elif 'open google' in self.query:
                    wb.open("google.com")
                elif 'open facebook' in self.query:
                    wb.open("facebook.com")
                elif 'face recognition' in self.query:
                    cmd = "python FaceRecognition.py"
                    p = sb.Popen(cmd, shell=True)
                elif 'create user' in self.query:
                    cmd = "python UserCreation.py"
                    p = sb.Popen(cmd, shell=True)
                    cmd = "python ModelTrainer.py"
                    p = sb.Popen(cmd, shell=True)
                elif 'country vaccination prediction' in self.query:
                    A = input("Enter date of prediction as DD-MM-YYYY")
                    
                    # Reading the dataset
                    ds = pd.read_csv("country_vaccinations.csv")
                    ds = ds[["iso_code", "date", "total_vaccinations"]]
                    ds = ds.dropna() #removes the values consisting NA

                    Idata = ds.loc[ds["iso_code"] == "IND"]
                    print(Idata)

                    dates = Idata[["date"]]
                    y1 = Idata[["total_vaccinations"]]

                    #function to covert the date into the number of days
                    def days(DATE):
                        dateNum = ""
                        y, m, d = None, None, None
                        num = "1234567890"

                        for i in DATE:
                            if i in num:
                                dateNum += i
                            elif i == "-":
                                if y is None:
                                    y = int(dateNum)
                                    dateNum = ""
                                elif m is None:
                                    m = int(dateNum)
                                    dateNum = ""
                            d = dateNum
                        return (date(y, m, int(d)) - date(2021, 1, 15)).days #Returns the days which are between 15jan to date

                    x1 = pd.DataFrame(dates["date"].apply(days))

                    #splitting the data into training and testing
                    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3,random_state=5)

                    #training the linear regression model 
                    model = LinearRegression()
                    model.fit(x_train, y_train)
                    y_pred1 = model.predict(x1)
                    print("Mean squared error =", metrics.mean_squared_error(y1, y_pred1))
                    print("score =", model.score(x_test, y_test) * 100, "%")
                    A=input("enter date(yyyy-mm-dd):")
                    #predicting using the model
                    result = model.predict([[days(A)]])
                    print("Total vaccinations by the given date =", result[0, 0])
                elif 'predict Wine quality' in self.query:

                    # Reading the dataset
                    ds = pd.read_csv ('winequality-red.csv')

                    #declaring X & Y
                    X = ds[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
                    Y = ds[['quality']]

                    #training the multiple regression model 
                    model = LinearRegression()
                    model.fit(X, Y)
                    print("['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']")
                    A=input("Enter the data in the above format to predict wine quality: ")

                    #predicting using the model
                    Y_pred = model.predict(A)
                    print("predicted value is: "+ Y_pred)

                elif 'cluster the wine data' in self.query:
                    
                    # Reading the dataset
                    df = pd.read_csv("Wine.csv")
                    df.head()

                    #Reducucing data to scale of 0 & 1 for higher accuracy
                    scaler = MinMaxScaler()
                    scaler.fit(df[0:14])
                    df[0:14] = scaler.transform(df[0:14])

                    N = input("number of clusters u want to take: ")

                    #Buliding the model
                    km = KMeans(n_clusters = N)  #for more features or groups we should take more clusters
                    y_predict = km.fit_predict(df[['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline', 'Customer_Segment']])


                    #Adding cluster data into the dataframe
                    df ['cluster'] = y_predict
                    print(df)

                elif 'predict diamond Price' in self.query:

                    # Reading the dataset
                    data = pd.read_csv("diamonds.csv")

                    #Rather than removing string features i am assigning or Replacing it with integers
                    data.replace({'Ideal': 1, 'Premium': 2, 'Good': 3, 'Very Good': 4, 'Fair': 5, 'E': 1, 'I': 2, 'J': 3, 'H': 4, 'G': 5, 'D': 6, 'F': 7, 'SI1': 1, 'SI2': 2, 'VS1': 3, 'VS2': 4, 'VVS1': 5, 'VVS2': 6, 'I1': 7, 'I2': 8, 'IF': 9}, inplace=True)

                    #allocating the data into x & y
                    x=data.iloc[:, [1, 2, 3, 4, 5, 6, 8, 9, 10]]
                    y=data.iloc[:, 7]
                    # print(x)
                    # print(y)
                    sc_x = StandardScaler()
                    x = sc_x.fit_transform(x)
                    N =input("The data for which the prediction should be done: ")
                    N = sc_x.transform(N)

                    #Buliding and Training the model
                    classifier = KNeighborsClassifier(n_neighbors = 3, p=2, metric='euclidean')

                    classifier.fit(x, y)
                    y_pred = classifier.predict(N)
                    print (y_pred)

startExecution = Mainthread()

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Pheonix()
        self.ui.setupUi(self)
        
        self.ui.pushButton.clicked.connect(self.startTask)
        self.ui.pushButton_2.clicked.connect(self.close)
        self.ui.pushButton_3.clicked.connect(self.camera)
        self.ui.pushButton_4.clicked.connect(self.google)
        self.ui.pushButton_5.clicked.connect(self.Youtube)
        self.ui.pushButton_6.clicked.connect(self.facebook)
    def camera(self):
        cmd = "python FaceRecognition.py"
        p = sb.Popen(cmd, shell=True)
    def google(self):
        wb.open("google.com")
    def Youtube(self):
        wb.open("youtube.com")
    def facebook(self):
        wb.open("facebook.com")
    

    def startTask(self):
        # make shure that the location saved is correct
        self.ui.movie = QtGui.QMovie("GIF/39f6a005763b37e2237b320df0e68e31.gif")
        self.ui.label_3.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie("GIF/tumblr_nrqm32yH3W1r6xm5co1_1280.gif")
        self.ui.label_5.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie("GIF/9e759fd37ccd98da121b670249f34afa.gif")
        self.ui.label_2.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie("GIF/tumblr_o7vrxl8Uk11runoqyo2_540.gif")
        self.ui.label_6.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie("GIF/dc8436d03012b7204637ed7207a05808.gif")
        self.ui.label.setMovie(self.ui.movie)
        self.ui.movie.start()
        startExecution.start()

app = QApplication(sys.argv)
Pheonix = Main()
Pheonix.show()
exit(app.exec_())