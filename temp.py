# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import print_function
#import potrebnih biblioteka
#%matplotlib inline
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import collections

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12 # za prikaz većih slika i plotova, zakomentarisati ako nije potrebno

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        #plt.figure()
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
def select_roi(image_orig, image_bin):
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
        if area > 100 and h < 100 and h > 15 and w > 20:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y+h+1,x:x+w+1]
            regions_array.append([resize_region(region), (x,y,w,h)])       
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions
def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255. 
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255
def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
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

def get_line_coords(frame):
    lines = cv2.HoughLinesP(frame, rho=1, theta=1 * np.pi /
                            180, threshold=100, minLineLength=100, maxLineGap=5)
    return lines

def calculate_line_params(line):
    x1, y1, x2, y2 = line
    m = (y2 - y1) / (x2 - x1)
    b = -m * x1 + y1
    return m, b

def get_lines_and_params(lines):
    ret = {}
    for i, lin in enumerate(lines):
        for x1, y1, x2, y2 in lin:
            m, b = calculate_line_params([x1, y1, x2, y2])
            ret[i] = [[m, b], [x1, y1, x2, y2]]
    return ret

def get_koord(lines):
    x1=lines[0][0][0]
    y1=lines[0][0][1]
    x2=lines[0][0][2]
    y2=lines[0][0][3]
    for line in lines:
        if line[0][0]<x1:
            x1=line[0][0]
        if line[0][1]>y1:
            y1=line[0][1]
        if line[0][2]>x2:
            x2=line[0][2]
        if line[0][3]<y2:
            y2=line[0][3]
            
    return x1,y1,x2,y2

def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def select_roi(image_orig, image_bin):
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
        if  h < 50 and h > 5 and w > 5:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y+h+1,x:x+w+1]
            regions_array.append([resize_region(region), (x,y,w,h)])       
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions
class Tacka:
    def __init__(self, x1,y1,x2,y2):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
    
# ucitavanje videa
frame_num = 0
cap = cv2.VideoCapture("video-8.avi")
cap.set(1, frame_num) # indeksiranje frejmova
# analiza videa frejm po frejm
#while True:
    #frame_num += 1
ret_val, frame = cap.read()
    # plt.imshow(frame)
    # ako frejm nije zahvacen
    #if not ret_val:
     #   break
#print(frame_num)
kernel = np.ones((3, 3)) # strukturni element 3x3 blok
display_image(frame)
plt.figure()
    # dalje se sa frejmom radi kao sa bilo kojom drugom slikom, npr
pom=frame[:,:,0]
#display_image(pom)
#plt.figure()

#frame_gray = cv2.cvtColor(pom, cv2.COLOR_BGR2GRAY)

#img_gray = cv2.cvtColor(pom, cv2.COLOR_RGB2GRAY)
#pom1=pom>120

pom1=cv2.erode(pom, kernel, iterations=1)
display_image(pom1)
plt.figure()
lines = cv2.HoughLinesP(pom1,1,np.pi/180,60,50,50)

x1,y1,x2,y2=get_koord(lines)
sab=Tacka(x1,y1,x2,y2)
linija=get_lines_and_params(lines)
print(linija)
print(lines)
print(sab.x1,sab.y1,sab.x2,sab.y2)
pom2=frame[:,:,1]
pom3=cv2.erode(pom2, kernel, iterations=1)
display_image(pom3)
lines = cv2.HoughLinesP(pom3,1,np.pi/180,60,50,50)

minus_x1,minus_y1,minus_x2,minus_y2=get_koord(lines)
print(minus_x1,minus_y1,minus_x2,minus_y2)
#print(lines[0][0][2])
#for line in lines:
    #print(line)
    #print(line[0][0])
    #print(line[1])

#print(lines)
#for x1,y1,x2,y2 in lines[0]:
    #print(x1,y1,x2,y2)
    #cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)


while True:
    frame_num += 1
    ret_val, frame = cap.read()
    
    if frame_num==50:
        break

    if not frame_num %3==0:
        continue
        
   # display_image(frame)
    #plt.figure()
    if not ret_val:
        break

print(frame_num)    
cap.release()
#return sum_of_nums



#image_color = load_image('images/brojevi.png')
img = invert(image_bin(image_gray(frame)))
#img_bin = erode(dilate(img))
selected_regions, numbers = select_roi(frame.copy(), img)
display_image(selected_regions)

for reg in numbers:
   display_image(reg)
   plt.figure()






