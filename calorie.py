from eval import eval
import cv2
import numpy as np 
from image_segment import *


density_dict = {'Apple':0.609, 'Banana':0.94, 'Carrot':0.641, 'Cucumber':0.641, 'Onion':0.513, 'Orange':0.482, 'Tomato':0.481}
calorie_dict = {'Apple':52, 'Banana':89, 'Carrot':41, 'Cucumber':16, 'Onion':40, 'Orange':47, 'Tomato':18}

#skin of photo to real multiplier
skin_multiplier = 5*2.3

def getCalorie(label, volume): #volume in cm^3
	calorie = calorie_dict[label]
	density = density_dict[label]
	mass = volume*density*1.0
	calorie_tot = (calorie/100.0)*mass
	return mass, calorie_tot, calorie #calorie per 100 grams

def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
	area_fruit = (area/skin_area)*skin_multiplier #area in cm^2

	volume = 100
	if label == 'Apple' or label == 'Tomato' or label == 'Orange' or label == 'Onion' : #sphere-apple,tomato,orange,kiwi,onion
		radius = np.sqrt(area_fruit/np.pi)
		volume = (4/3)*np.pi*radius*radius*radius
		#print (area_fruit, radius, volume, skin_area)
	
	if label == 'Banana' or label == 'Cucumber' or (label == 'Carrot' and area_fruit > 30): #cylinder like banana, cucumber, carrot
		fruit_rect = cv2.minAreaRect(fruit_contour)
		height = max(fruit_rect[1])*pix_to_cm_multiplier
		radius = area_fruit/(2.0*height)
		volume = np.pi*radius*radius*height
		
	if (label=='Cucumber' and area_fruit < 30) : # carrot
		volume = area_fruit*0.5 #assuming width = 0.5 cm
	
	return volume

def calories(result,img):
	img_path =img # "C:/Users/M Sc-2/Desktop/dataset/FooD/"+str(j)+"_"+str(i)+".jpg"
	fruit_areas, final_f, areaod, skin_areas, fruit_contours, pix_cm = getAreaOfFood(img_path)
	volume = getVolume(result, fruit_areas, skin_areas, pix_cm, fruit_contours)
	mass, cal, cal_100 = getCalorie(result, volume)
	fruit_volumes=volume
	fruit_calories=cal
	fruit_calories_100grams=cal_100
	fruit_mass=mass
	#print("\nfruit_volumes",fruit_volumes,"\nfruit_calories",fruit_calories,"\nruit_calories_100grams",fruit_calories_100grams,"\nfruit_mass",fruit_mass)
	return fruit_calories