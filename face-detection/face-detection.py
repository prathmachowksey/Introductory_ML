
import cv2 as cv

# Read image from your local file system
original_image = cv.imread('./faces2.jpeg')

# Convert color image to grayscale for Viola-Jones
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

#Haar Cascade for face detection
face_cascade = cv.CascadeClassifier('./opencvfiles/haarcascade_frontalface_alt.xml')



detected_faces = face_cascade.detectMultiScale(grayscale_image,scaleFactor=1.3,minNeighbors=5)

for (column, row, width, height) in detected_faces: #x,y,w,h


	cv.rectangle(
		original_image,
		(column, row),#x,y of top left corner
		(column + width, row + height), #x,y of bottom right corner
		(0, 255, 0),  #color in rgb
		2 #line thickness
	)


cv.imshow('Image', original_image)
cv.waitKey(0) #so that image window does not close until a key is pressed
cv.destroyAllWindows()


#This works only for frontal faces. Notice how all the faces in the file 'faces.jpeg' are identified, but not those in the file 'faces2.jpeg'
