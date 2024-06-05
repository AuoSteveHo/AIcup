#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os 
import shutil
import numpy as np


def image_process(img, alpha, beta):
        img = np.float32(img)
        img = cv2.multiply(img, np.array([alpha]))
        img = cv2.add(img, beta)
        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        #img = cv2.equalizeHist(img)  
        #clahe = cv2.createCLAHE (clipLimit = 2.0, tileGridSize=(10, 10))  
        #img = clahe.apply(img) 
        img = cv2.fastNlMeansDenoisingColored(img, None, 3,3,7,21)
        return img

def create_txtFile(filename):
    with open(filename, 'w+') as f:
        pass
		
def write_newTxt(filename, word):
    with open(filename, 'a') as r:
        r.write(word[0]+" ")
        r.write(str(round(word[1],6))+" ")
        r.write(str(round(word[2],6))+" ")
        r.write(str(round(word[3],6))+" ")
        r.write(str(round(word[4],6))+" ")
        r.write(str(round(word[5],6))+"\n")


min_alpha = 0.5
max_alpha = 1.5
min_beta = 0
max_beta = 30
alpha = 0
beta = 0

folder_path = './augmentation/'
allFolderList = os.listdir(folder_path)

for sub_folder in allFolderList:
    sub_folder_path = os.path.join(folder_path, sub_folder)
    if os.path.isdir(sub_folder_path):
        print(sub_folder_path)
        files = os.listdir(sub_folder_path)
        for file in files:
            alpha = min_alpha
            beta = min_beta
            if file.endswith('.jpg'):
                print(file)
                image = cv2.imread(sub_folder_path+'/'+file)
                txt_file = file.replace(".jpg", ".txt")
                print(txt_file)

                while alpha <= max_alpha:
                    while beta <= max_beta:
                        img = image_process(image, alpha, beta)
                        img2 = cv2.flip(img, 1)
                        cv2.imwrite(sub_folder_path+'/'+str(int(10*alpha))+"_"+str(beta)+"a"+file, img)
                        cv2.imwrite(sub_folder_path+'/'+str(int(10*alpha))+"_"+str(beta)+"b"+file, img2)
                        create_txtFile(sub_folder_path+'/'+str(int(10*alpha))+"_"+str(beta)+"a"+txt_file)
                        create_txtFile(sub_folder_path+'/'+str(int(10*alpha))+"_"+str(beta)+"b"+txt_file)
                        with open(sub_folder_path+'/'+txt_file) as t:
                            for line in t.readlines():
                                word = line.split(" ")
                                word_a = [word[0], float(word[1]), float(word[2]), float(word[3]), float(word[4]), int(word[5])]
                                word_b = [word[0], 1-float(word[1]), float(word[2]), float(word[3]), float(word[4]), int(word[5])]
                                write_newTxt(sub_folder_path+'/'+str(int(10*alpha))+"_"+str(beta)+"a"+txt_file, word_a)
                                write_newTxt(sub_folder_path+'/'+str(int(10*alpha))+"_"+str(beta)+"b"+txt_file, word_b)
                        beta+=15
                    beta = min_beta
                    alpha += 0.5
                    

