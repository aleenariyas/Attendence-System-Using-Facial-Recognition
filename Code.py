#FacePool-An attendance system which uses facial recognition
#FacePool uses the OpenCV library for face detection, Tkinter for the graphical user interface, and pandas for data manipulation.
#vggg

#Importing necessary libraries (Packages) and modules:
import tkinter as tk
from tkinter import * #for GUI
from screeninfo import get_monitors #for retrieving monitor information
import cv2,os #cv2 for computer vision tasks (OpenCV)
import csv #for handling data
import numpy as np #for numerical computing
import pandas as pd #for handling data
import datetime
import time
from PIL import Image #Python library for image processing tasks
import glob #for searching files
from pandastable import Table #provides a table widget



#Setting up the GUI
window = Tk()
window.title("FacePool")
alphaerror = tk.PhotoImage(file = f"Asset\\alphaerror.png")
numerror = tk.PhotoImage(file = f"Asset\\numerror.png")
invalidentry = tk.PhotoImage(file = f"Asset\\invalidentry.png")
trained = tk.PhotoImage(file = f"Asset\\trained.png")
imagesaved = tk.PhotoImage(file = f"Asset\\imagesaved.png")
done = tk.PhotoImage(file = f"Asset\\done.png")



def is_number(s): #Checks if a string is a valid number
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False




#Captures images using the computer's webcam and saving them along with some user details for training the facial recognition model
def TakeImages():
    global alphaerror, numerror, trained, done, invalidentry
    Id = (entry0.get())
    name = (entry1.get())
    if (is_number(Id) and name.isalpha()): #checks if the entered ID is a number and if the entered name consists only of alphabetic characters
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('frame', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 60
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        row = [Id, name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        c5 = canvas.create_image(276, 522, image=imagesaved)
        canvas.after(6000, lambda: canvas.itemconfig(c5, state='hidden'))
    else: #check for error in entered info
        if (is_number(Id)):
            c1 = canvas.create_image(287.0, 480.0, image=alphaerror)
            canvas.after(6000, lambda: canvas.itemconfig(c1, state='hidden'))

        if (name.isalpha()):
            c2 = canvas.create_image(287.0, 480.0, image=numerror)
            canvas.after(6000, lambda: canvas.itemconfig(c2, state='hidden'))

        if (name.strip() == "" and Id.strip() == ""):
            c3 = canvas.create_image(287.0, 480.0, image=invalidentry)
            canvas.after(6000, lambda: canvas.itemconfig(c3, state='hidden'))





def TrainImages(): #performs face recognition training using the LBPH (Local Binary Patterns Histograms) algorithm
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml" #path to the XML file containing the Haar cascade for face detection
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml") #saves the trained recognizer model to a file named "Trainner.yml"
    c4 = canvas.create_image(276, 522, image=trained)
    canvas.after(6000, lambda: canvas.itemconfig(c4, state='hidden'))





#retrieves the images and corresponding labels from a specified directory
def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empty face list
    faces = []
    # create empty  list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids




#performs real-time face recognition and attendance tracking using a trained model
def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")   #loads the trained model
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);  #initializes the face cascade classifier
    df = pd.read_csv("StudentDetails\StudentDetails.csv")  #reads the student details
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  #initializes the video capture
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names) #DataFrame
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):  #If the confidence of the prediction is less than 50, it considers the face recognized
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%I:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id) + "-" + aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]

            else:
                Id = 'Unknown'
                tt = str(Id)
            if (conf > 75): #If the confidence is greater than 75, indicating an unknown face
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%I:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance\Attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    c6 = canvas.create_image(276, 522, image=done)
    canvas.after(6000, lambda: canvas.itemconfig(c6, state='hidden'))




#retrieves attendance data from CSV files in a specific directory and displays it in a table format
def printdata():

    # defining root path
    path = '.'
    filenames = glob.glob("D:\FacePool\Attendance" + "/*.csv")
    # creating empty list
    dataFrames = list()
    # iterating through CSV file in current directory
    for filename in filenames:
        dataFrames.append(pd.read_csv(filename))
    # Concatenate all data into one DataFrame
    merged_frame = pd.concat(dataFrames, axis=1)
    all_files = glob.glob(os.path.join("D:\FacePool\Attendance", "*.csv"))
    df1 = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    df1.sort_values("Time",ascending=False, inplace=True)
    df1.drop_duplicates(subset='Id', keep='first', inplace=True) #DropDupicate
    df = pd.DataFrame(df1)
    extra_window = tk.Toplevel()
    extra_window.title('Attendance')
    frame = tk.Frame(extra_window)
    frame.pack()
    pt = Table(frame, width=1100, height=618, dataframe=df, showtoolbar=True, showstatusbar=True)
    pt.show()
    WIN_WIDTH = 1100
    WIN_HEIGHT = 618
    extra_window.geometry(
        f"{WIN_WIDTH}x{WIN_HEIGHT}+{(get_monitors()[0].width - WIN_WIDTH) // 2}+{(get_monitors()[0].height - WIN_HEIGHT) // 2}")
    extra_window.iconbitmap(f"Asset\icon.ico")
    extra_window.mainloop()




#displays an "About FacePool" window
def about():
    extra_window2 = tk.Toplevel()
    WIN_WIDTH = 923
    WIN_HEIGHT = 518
    extra_window2.geometry(
        f"{WIN_WIDTH}x{WIN_HEIGHT}+{(get_monitors()[0].width - WIN_WIDTH) // 2}+{(get_monitors()[0].height - WIN_HEIGHT) // 2}")
    extra_window2.configure(bg="#FFFFFF")
    extra_window2.title("About FacePool")
    canvas1 = Canvas(
        extra_window2,
        bg="#FFFFFF",
        height=518,
        width=923,
        bd=0,
        highlightthickness=0,
        relief="ridge")
    canvas1.place(x=0, y=0)
    background_img1 = PhotoImage(file=f"Asset\sbg.png")
    bg = canvas1.create_image(
        511.5, 278.0,
        image=background_img1)
    extra_window2.iconbitmap(f"Asset\icon.ico")
    extra_window2.resizable(False, False)
    extra_window2.mainloop()



#GUI SETUP
window.iconbitmap(f"Asset\icon.ico")
WIN_WIDTH = 1100
WIN_HEIGHT = 618
window.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}+{(get_monitors()[0].width - WIN_WIDTH)//2}+{(get_monitors()[0].height - WIN_HEIGHT)//2}")
window.configure(bg = "#ffffff")

canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 618,
    width = 1100,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"Asset\\background.png")
background = canvas.create_image(
    550.0, 315.0,
    image=background_img)

img0 = PhotoImage(file = f"Asset\img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = TrackImages,
    relief = "flat")

b0.place(
    x = 708, y = 218,
    width = 210,
    height = 43)

img1 = PhotoImage(file = f"Asset\img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = printdata,
    relief = "flat")

b1.place(
    x = 708, y = 298,
    width = 210,
    height = 43)

img2 = PhotoImage(file = f"Asset\img2.png")
b2 = Button(
    image = img2,
    borderwidth = 0,
    highlightthickness = 0,
    command = TrainImages,
    activebackground="#1269C0",
    relief = "flat")

b2.place(
    x = 327, y = 377,
    width = 110,
    height = 32)

img3 = PhotoImage(file = f"Asset\img3.png")
b3 = Button(
    image = img3,
    borderwidth = 0,
    highlightthickness = 0,
    command = TakeImages,
    activebackground="#0D5FB1",
    relief = "flat")

b3.place(
    x = 136, y = 381,
    width = 110,
    height = 30)

entry0_img = PhotoImage(file = f"Asset\img_textBox0.png")
entry0_bg = canvas.create_image(
    287.5, 235.0,
    image = entry0_img)

entry0 = Entry(
    bd = 0,
    bg = "#eaf4ff",
    highlightthickness = 0)

entry0.place(
    x = 144.0, y = 218,
    width = 287.0,
    height = 32)

entry1_img = PhotoImage(file = f"Asset\img_textBox1.png")
entry1_bg = canvas.create_image(
    287.5, 320.0,
    image = entry1_img)

entry1 = Entry(
    bd = 0,
    bg = "#eaf4ff",
    highlightthickness = 0)

entry1.place(
    x = 144.0, y = 303,
    width = 287.0,
    height = 32)

img4 = PhotoImage(file = f"Asset\img4.png")
b4 = Button(
    image = img4,
    borderwidth = 0,
    highlightthickness = 0,
    command = about,
    relief = "flat")

b4.place(
    x = 1054, y = 570,
    width = 20,
    height = 21)

window.resizable(False, False)
window.mainloop()

#END OF CODE
