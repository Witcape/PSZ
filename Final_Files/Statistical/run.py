import cv2
import numpy as np
from bounding_fun import bounding_function

# Load an image
I = cv2.imread("/Users/shauryabhardwaj/Desktop/ML/Zero_Shot/Aupendu/zero-restore/I-HazeFullx/Input/Indoor_31_N18.png")  
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)  

zeta = 0.5  # Example parameter value
r, est_tr_proposed, A = bounding_function(I, zeta)

# Save or display results
cv2.imwrite("output.jpg", (r * 255).astype(np.uint8))
cv2.imshow("Dehazed Image", (r * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
