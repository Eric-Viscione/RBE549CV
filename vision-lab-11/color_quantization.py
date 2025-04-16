import numpy as np
import cv2 as cv
libraries = True
try:
    from shared_libraries import stack_images, load_images, standard_show

except ImportError:
    print("Optional module not found.")
    libraries = False
if libraries:
    img = load_images(["nature.png"])
else:
    img = cv.imread("nature.png")
Z = img.reshape((-1,3))
 
# convert to np.float32
results = []

Z = np.float32(Z)
 
# define criteria, number of clusters(K) and apply kmeans()
ks = [2, 3, 5, 10, 20, 40]
imgs = []
imgs.append(img)
for K in ks:
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    ret,label,center=cv.kmeans(Z,K,None,criteria,20,cv.KMEANS_PP_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    text = f"K value: {K}"
    cv.putText(res2, text, (50,50), cv.FONT_HERSHEY_SIMPLEX, 2, color = (255, 255, 255), thickness = 2 )

    imgs.append(res2)
if libraries:
    standard_show(stack_images(imgs, 3), save=True)

# cv.imshow('res2',res2)
# cv.waitKey(0)
# cv.destroyAllWindows()