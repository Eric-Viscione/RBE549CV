import numpy as np
import cv2 as cv
import os


def stack_images(images, cols):
    rows = (len(images) + cols - 1) // cols  
    size = (300, 300) 
    imgs_resized = [cv.resize(img, size) for img in images]
    empty = np.zeros_like(imgs_resized[0])
    imgs_resized += [empty] * (rows * cols - len(images))  
    spacing_line = np.zeros((size[1], 25), dtype=np.uint8) 
    
    stacked_images = []

    for i in range(rows):
        row = np.hstack(imgs_resized[i * cols:(i + 1) * cols]) 
        stacked_images.append(row)
    return np.vstack(stacked_images)


class Image_transforms:
    
    def __init__(self):
        
        pass
    def add_label(self, img, label):
        font = cv.FONT_HERSHEY_SIMPLEX
        rows,cols, _ = img.shape
        bottom_middle_x = (cols // 2)- 150
        bottom_middle_y = rows - 5
        cv.putText(img,label, (bottom_middle_x, bottom_middle_y), font, 1, (0, 0, 0), 2)
        return img
    def scale_image(self, img, scale):
        height, width = img.shape[:2]
        res = cv.resize(img,(int(scale*width), int(scale*height)), interpolation = cv.INTER_CUBIC)
        res = self.add_label(res, f"Scaled Image {scale*100}%")
        return res
    
    def translate(self, img, M = ([[1,0,100],[0,1,50]])):
        rows,cols, _ = img.shape
        dst = cv.warpAffine(img,M,(cols,rows))
        dst = self.add_label(dst, "Translated Image")
        return dst
    
    def rotate(self, img, angle):
        rows,cols, _ = img.shape
        M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),angle,1)
        dst = cv.warpAffine(img,M,(cols,rows))
        dst = self.add_label(dst, f"Rotated Image {angle}deg")

        return dst
    def affine(self,img):
        rows,cols, _ = img.shape
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        M = cv.getAffineTransform(pts1,pts2)
        dst = cv.warpAffine(img,M,(cols,rows))
        dst = self.add_label(dst, "Affine Transform")
        return dst
    def perspective(self, img):
        pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
        pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
        M = cv.getPerspectiveTransform(pts1,pts2)
        dst = cv.warpPerspective(img,M,(300,300))
        dst = self.add_label(dst, "Perspective Transform")
        return dst
class Image_classifiers:
    def __init__(self):
        pass
    def harris(self, img, blocksize=2, ksize=3, k=0.04, thresh=0.05):
        print(type(img))
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        dst = cv.cornerHarris(gray_img,blocksize, ksize, k)
        img_copy = img.copy()
        img_copy[dst > thresh * dst.max()] = [0, 0, 255]
        return img_copy

    def sift(self,img, nfeatures=0, nOctavelayers=3, constant_threshold=0.1, edge_threshold=10, sigma=1.6 ):
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create(nfeatures=nfeatures, 
                                   nOctaveLayers=nOctavelayers, 
                                   contrastThreshold=constant_threshold, 
                                   edgeThreshold=edge_threshold, 
                                   sigma=sigma)

        keypoints, descriptors = sift.detectAndCompute(gray, None)
        dst = cv.drawKeypoints(gray, keypoints, img, color = (0, 255, 0))
        return dst



def main():


    ##rotation by 10deg
    ##scale up by 20%
    ##scale doiwn by 20%
    ##Affine Transformation
    ##Perspective Transformation
    classifiers = Image_classifiers()
    transformer = Image_transforms()
    img = cv.imread('unity_hall.jpg')
    assert img is not None, "file could not be read, check with os.path.exists()"
    rotated = transformer.rotate(img, 10)
    scaled_up20 =  transformer.scale_image(img, 1.2)
    scaled_down20 = transformer.scale_image(img, 0.8)
    affine = transformer.affine(img)
    perspective = transformer.perspective(img)
    images = [img, rotated, scaled_up20, scaled_down20, affine, perspective]
    images_identified = []
    for image in images:
        harris = classifiers.harris(image)
        sift = classifiers.sift(image)
        images_identified.append(harris)
        images_identified.append(sift)
    # stacked_img = stack_images(images, 3)
    # cv.imshow("Stacked Images", stacked_img)
    # harris = classifiers.harris(img)
    # sift = classifiers.sift(img)
    stacked_img = stack_images(images_identified, 3)
    cv.imshow("Stacked Images", stacked_img)
    # # cv.imshow("Harris", harris)
    # cv.imshow("sift", sift)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
 

if __name__ == "__main__":
    main()
 