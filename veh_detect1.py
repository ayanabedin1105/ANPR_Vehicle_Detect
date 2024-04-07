import cv2
import numpy as np

# Define minimum width and height for contours
min_contour_width = 40  
min_contour_height = 40  

# Define offset for detecting vehicles crossing a line
offset = 10  

# Define the height of the line for vehicle counting
line_height = 550  

# List to store centroid coordinates of detected objects
matches = []

# Counter for the total number of vehicles detected
vehicles = 0

# Function to calculate centroid of a bounding box
def get_centroid(x, y, w, h):
   x1 = int(w / 2)
   y1 = int(h / 2)
 
   cx = x + x1
   cy = y + y1
   return cx, cy

# Open video file for processing
cap = cv2.VideoCapture('Video.mp4')

# Set video width and height
cap.set(3, 1920)
cap.set(4, 1080)

# Read the first frame from the video
if cap.isOpened():
   ret, frame1 = cap.read()
else:
   ret = False
ret, frame1 = cap.read()

# Read the second frame from the video
ret, frame2 = cap.read()

# Loop through video frames
while ret:
   # Compute the absolute difference between frames
   d = cv2.absdiff(frame1, frame2)
   grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

   # Apply Gaussian blur to reduce noise
   blur = cv2.GaussianBlur(grey, (5, 5), 0)

   # Apply thresholding to get binary image
   ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

   # Perform dilation to fill gaps in the image
   dilated = cv2.dilate(th, np.ones((3, 3)))

   # Define kernel for morphological operations
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

   # Perform closing operation to close small holes
   closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

   # Find contours in the binary image
   contours, h = cv2.findContours(
       closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
   # Loop through each contour
   for(i, c) in enumerate(contours):
       # Get bounding box coordinates
       (x, y, w, h) = cv2.boundingRect(c)
       
       # Check if contour is valid based on minimum width and height
       contour_valid = (w >= min_contour_width) and (h >= min_contour_height)

       if not contour_valid:
           continue
       
       # Draw bounding box around the contour
       cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)

       # Draw a line for vehicle counting
       cv2.line(frame1, (0, line_height), (1200, line_height), (0, 255, 0), 2)
       
       # Get centroid of the bounding box
       centroid = get_centroid(x, y, w, h)
       
       # Add centroid to the list of matches
       matches.append(centroid)
       
       # Draw a circle at the centroid
       cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
       
       # Check if the centroid crosses the counting line
       cx, cy = get_centroid(x, y, w, h)
       for (x, y) in matches:
           if y < (line_height+offset) and y > (line_height-offset):
               # Increment vehicle count if centroid crosses the line
               vehicles = vehicles + 1
               # Remove centroid from matches list
               matches.remove((x, y))

   # Display total number of vehicles detected
   cv2.putText(frame1, "Total Vehicle Detected: " + str(vehicles), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
               (0, 170, 0), 2)

   # Display the frame
   cv2.imshow("Vehicle Detection", frame1)
   
   # Wait for key press and check if 'Esc' is pressed
   if cv2.waitKey(1) == 27:
       break
   
   # Update frame1 and frame2 for next iteration
   frame1 = frame2
   ret, frame2 = cap.read()

# Destroy all OpenCV windows and release video capture
cv2.destroyAllWindows()
cap.release()
