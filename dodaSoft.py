# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:35:28 2019

@author: Dejan Doder
"""
from __future__ import print_function
#import potrebnih biblioteka
#%matplotlib inline
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import collections
import argparse
import math
from skimage import morphology

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.models import model_from_json

import matplotlib.pylab as pylab
#pylab.rcParams['figure.figsize'] = 16, 12 

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    #image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

#def image_bin(image_gs):
#    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
#    return image_bin

#Funkcionalnost implementirana u OCR basic
def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized
def scale_to_range(image):
    return image / 255
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona 
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
        
    return ready_for_ann
def convert_output(alphabet):
    '''Konvertovati alfabet u niz pogodan za obučavanje NM,
        odnosno niz čiji su svi elementi 0 osim elementa čiji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)
def winner(output): # output je vektor sa izlaza neuronske mreze
    '''pronaći i vratiti indeks neurona koji je najviše pobuđen'''
    return max(enumerate(output), key=lambda x: x[1])[0]
def display_result(outputs, alphabet):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def my_rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))  # zauzimanje memorije za sliku (nema trece dimenzije)
    img_gray = 0.21*img_rgb[:, :, 0] + 0.77*img_rgb[:, :, 1] + 0.07*img_rgb[:, :, 2]
    img_gray = img_gray.astype('uint8')  # u prethodnom koraku smo mnozili sa float, pa sada moramo da vratimo u [0,255] opseg
    return img_gray

def create_ann():
    '''Implementacija veštačke neuronske mreže sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann
    
def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''
    X_train = np.array(X_train, np.float32) # dati ulazi
    y_train = np.array(y_train, np.float32) # zeljeni izlazi za date ulaze
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=2000, batch_size=1, verbose = 0, shuffle=False) 
      
    return ann

def select_roi(image_orig, image_bin, linija):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28. 
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    x1=linija[0][0]
    y1=linija[0][1]
    x2=linija[0][2]
    y2=linija[0][3]
    
    #detektovan=False
    brojac=0
    #print('iz fje',x1,y1,x2,y2)
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    lista=[]
    
   
    for contour in contours: 
        
        x,y,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        
        
        if h < 50 and h > 10 and w > 5 and area>100:
               brojac+=1
               distanca, blizina = pnt2line((x,y,0), (x1,y1,0), (x2,y2,0))
               if distanca<3:
                   lista.append([distanca,blizina,x,y,w,h])
    sorted_lista=[]      
    if len(lista)>0:
        sorted_lista=sorted(lista,key=lambda item: item[0])
        distanca, blizina=sorted_lista[0][0], sorted_lista[0][1]
        x,y,w,h=sorted_lista[0][2],sorted_lista[0][3],sorted_lista[0][4],sorted_lista[0][5]
        region = image_bin[y:y+h+1,x:x+w+1]
        regions_array.append([resize_region(region), (x,y,w,h)])       
        cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
        regions_array = sorted(regions_array, key=lambda item: item[1][0])
        sorted_regions = sorted_regions = [region[0] for region in regions_array]
        #print(contours) 
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions

def select_roi1(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28. 
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 100 and h < 50 and h > 10 and w > 5:
            
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y+h+1,x:x+w+1]
            regions_array.append([resize_region(region), (x,y,w,h)])       
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions
def hist(image):
    height, width = image.shape[0:2]
    x = range(0, 256)
    y = np.zeros(256)
    
    for i in range(0, height):
        for j in range(0, width):
            pixel = image[i, j]
            y[pixel] += 1
    
    return (x, y)

def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z
  
def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)
  
def vector(b,e):
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)
  
def unit(v):
    x,y,z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)
  
def distance(p0,p1):
    return length(vector(p0,p1))
  
def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)
  
def add(v,w):
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)

#funkcija za detektovanje broja na liniji
#funkcija implementirana na osnovu vektora linije i vektora od pocetka linije do broja
def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t <= 0.0:
        t = 0.0
    elif t >= 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)
def koordinate_linija(linija):
    
    x1=linija[0]
    y1=linija[1]
    x2=linija[2]
    y2=linija[3]
     
    return (x1,y1,x2,y2)

# citanje obucenog modela i upis u fajl
    
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
ann = model_from_json(model_json)
ann.load_weights("model.h5")
    
file= open("out.txt","w+")
file.write("RA 220/2015 Dejan Doder\r")
file.write("file	sum\r")

#prolaz kroz video snimke i analiza frejm po frejm
    
for i in range(0,1):
    
    cap = cv2.VideoCapture('video/video-'+str(i)+'.avi')
    frame_num = 0
    cap.set(1, frame_num) # indeksiranje frejmova
    # analiza videa frejm po frejm
    #while True:
    #    frame_num += 1
    ret_val, frame = cap.read()
    
    siva = my_rgb2gray(frame)
    
    kernel = np.ones((3, 3))
    izdvajanje_crvene=frame[:,:,0]
    izdvajanje_zelene=frame[:,:,1]
    
    #display_image(izdvajanje_crvene)
    #plt.figure()
    
    erozija1=cv2.erode(izdvajanje_crvene, kernel, iterations=1)
    
    erozija2=cv2.erode(izdvajanje_zelene, kernel, iterations=1)
    
    ret, image_binarna = cv2.threshold(siva, 93, 255, cv2.THRESH_BINARY) # ret je vrednost praga, image_bin je binarna slika
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #print(kernel)
    img_ero = cv2.erode(image_binarna, kernel, iterations=1)
    img_open = cv2.dilate(img_ero, kernel, iterations=1)
    #plt.imshow(img_open, 'gray')
    #plt.figure()
    
    lines = cv2.HoughLinesP(img_open,1,np.pi/180,60,50,50)
    #print(lines)
    zelena_linija=lines[0]
    print('Zelena linija',zelena_linija)
    
    ret, image_bin1 = cv2.threshold(izdvajanje_crvene, 93, 255, cv2.THRESH_BINARY) # ret je vrednost praga, image_bin je binarna slika
    #print(ret)
    #plt.imshow(image_bin1, 'gray')
    
    img_ero1 = cv2.erode(image_bin1, kernel, iterations=1)
    img_open1 = cv2.dilate(img_ero1, kernel, iterations=1)
    #plt.imshow(img_open1, 'gray')
    #plt.figure()
    
    lines = cv2.HoughLinesP(img_open1,1,np.pi/180,60,50,50)
    #print(lines)
    crvena_linija=lines[0]
    print('Crvena linija',crvena_linija)
   # print(crvena_linija[0][0])
    
    x1,x2,y1,y2=koordinate_linija(crvena_linija[0])
    #print(x1,x2,y1,y2)
    
    ######################################################################
    #img_bw = 255*(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')
    #se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    #se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    #mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    #mask = np.dstack([mask, mask, mask]) / 255
    #out = frame * mask
    #display_image(out)
    #plt.figure()
    #frame1 = morphology.remove_small_objects(green, min_size=64, connectivity=2)
    #dst = cv2.fastNlMeansDenoisingColored(frame,None,10,1,7,10)
    #frame = rgb_image.copy() # Make a copy
    #frame[:,:,0] = 0
    #frame[:,:,2] = 0
    #frame1=morphology.remove_small_objects(frame, min_size=10, connectivity=5)
    #display_image(dst)
    #plt.figure()
    #image_color = load_image('images/brojevi.png')
    #denoised = cv2.fastNlMeansDenoising(
    #        gray, h=18, searchWindowSize=25, templateWindowSize=11)
    
    img = invert(image_bin(image_gray(frame)))
    img_bin = erode(dilate(frame))
    selected_regions, numbers = select_roi1(frame.copy(), img)

    broj_piksela=[0,0]
    broj_piksela_minus=[0,0]

    brojevi_crvena_linija=[]
    brojevi_zelena_linija=[]
    broj_crnih=0
    broj_bijelih=0
    rezultat=0
    broj=[]
    zbir_piksela=[]
    while True:
        frame_num += 1
        ret_val, frame = cap.read()
        
        if not ret_val:
            break
        
        
        
        binarna=image_bin(image_gray(frame))
        #selektovanje regiona koji prelaze preko crvene linije
        image_orig, num=select_roi(frame,binarna,crvena_linija)
        for reg in num:
            
            x,y=hist(reg)
            #print(reg[:][0])
#            if reg.all()==255:
#                broj_crnih+=broj_crnih
#            else:
#                broj_bijelih+=broj_bijelih
#            zbir_piksela=[broj_crnih,broj_bijelih]
#            print(zbir_piksela)
                
            
            for i in len(reg):
                for j in len(reg):
                    if(reg[i][j]==255):
                        broj_crnih+=broj_crnih
                    else:
                        broj_bijelih+=broj_bijelih
            zbir_piksela=[broj_crnih,broj_bijelih]
            print(zbir_piksela)
                
           
            broj_piksela.append(y[-1])
            if (abs(y[-1]-(broj_piksela[-1]))<10 or abs(y[-1]-(broj_piksela[-2]))<10 ):
                brojevi_crvena_linija.append(reg)
                display_image(reg)
                plt.figure()

        #selektovanje regiona koji prelaze preko zelene linije
        image_orig1, num1 = select_roi(frame, binarna, zelena_linija)
        
        for reg in num1:
            x,y=hist(reg)
            broj_piksela_minus.append(y[-1])
            if (abs(y[-1]-(broj_piksela_minus[-1]))<7 or abs(y[-1]-(broj_piksela_minus[-2]))<7):
                brojevi_zelena_linija.append(reg)
       
    cap.release()    
    
    alphabet = [0,1,2,3,4,5,6,7,8,9]
    niz_plus=[]
    niz_minus=[]
    
    if not (not brojevi_crvena_linija):
        rezultat_plus = ann.predict(np.array(prepare_for_ann(brojevi_crvena_linija),np.float32))
        niz_plus=display_result(rezultat_plus,alphabet)
        
    for broj in niz_plus:
        rezultat+=broj
        
    if not (not brojevi_zelena_linija):
        rezultat_minus = ann.predict(np.array(prepare_for_ann(brojevi_zelena_linija),np.float32))
        niz_minus=display_result(rezultat_minus,alphabet)
        
    for broj in niz_minus:
        rezultat-=broj
        
    print("Rezultat: ",rezultat)
    
    file.write('video-'+str(i)+'.avi\t' + str(rezultat)+'\r')

file.close()




