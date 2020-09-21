from eval import *
from calorie import *
from image_segment import *
import os,cv2,argparse
import matplotlib.pyplot as plt

IMG_SIZE = 800

parser = argparse.ArgumentParser()
parser.add_argument('--img' ,help="Image Path")
parser.add_argument('--model' ,help="Model path")

args=parser.parse_args()


image_path = os.path.abspath(args.img)
model_path = os.path.abspath(args.model)

#prediction = eval(image_path, model_path)
prediction = "Apple"
print("Predicted = ",prediction)

# prediction = "Apple"
img=cv2.imread(image_path)
img1=cv2.resize(img,(IMG_SIZE,IMG_SIZE))

threshold = 200
cat = " "

cal=round(calories(prediction,img),2)
if(threshold > cal):
	print(" Calories:",str(cal))
	print("Food Is Healthy")
	cat = "Healthy"
else:
	print(" Calories:",str(cal))
	print("Food Is UnHealthy")
	cat = "Unhealthy"
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title('{}  ({}kcal)  {}'.format(prediction,cal,cat))
plt.axis('off')
plt.show()



