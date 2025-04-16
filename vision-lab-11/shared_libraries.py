import numpy as np
import cv2 as cv
 

def stack_images(images, cols, scale_factor = .75):
    rows = (len(images) + cols - 1) // cols  
    # size = (300, 300) turn this on to resize to an arbitrary size
    # scale_factor = 0.5
    size = (int(images[0].shape[1]*scale_factor), int(images[0].shape[0]*scale_factor))  # (width, height)
    imgs_resized = [cv.resize(img, size) for img in images]
    empty = np.zeros_like(imgs_resized[0])
    imgs_resized += [empty] * (rows * cols - len(images))  
    spacing_line = np.zeros((size[1], 25), dtype=np.uint8) 

    stacked_images = []
    for i in range(rows):
        row = np.hstack(imgs_resized[i * cols:(i + 1) * cols]) 
        stacked_images.append(row)
    return np.vstack(stacked_images)





def load_images(names,channels=cv.IMREAD_COLOR):
    images = []
    for name in names:
        print(name)
        image = cv.imread(name, channels)
        if image is None:
            print(f'Could Not load image {name}. Check the path ')
            continue
        images.append(image)
    if not images:
        raise ValueError("No valid images were loaded. Please check the file paths.")
    if len(images) <= 1:
        image = images[0]
        return image
    return images
def standard_show(image, name="image", save = False):

    cv.imshow(name, image)
    if save:
        file = f"{name}.jpg"
        cv.imwrite(file, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def apply_kernel( frame, kernel, padding = True, fast_algo = True):
    """A generic function to apply a generic kernel to any generic image

    Args:
        frame (_type_): The input frame
        kernel (_type_): Any size kernel to be used
        padding (bool, optional): Add padding to the image to ensure it stays the same size
        fast_algo (bool, optional): This is a faster implementation of kernel actions, much faster than the default one. Defaults to True.

    Returns:
        _type_: _description_
    """
    height, width = frame.shape[0], frame.shape[1]
    kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]
    if fast_algo:   ##this implementation is inspired by https://stackoverflow.com/questions/64587303/python-image-convolution-using-numpy-only
        result_height, result_width = height - kernel_height + 1, width - kernel_width + 1
        ix0 = np.arange(kernel_height)[:, None] + np.arange(result_height)[None, :]
        ix1 = np.arange(kernel_width)[:, None] + np.arange(result_width)[None, :]
        res = kernel[:, None, :, None] * frame[(ix0.ravel()[:, None], ix1.ravel()[None, :])].reshape(kernel_height, result_height, kernel_width, result_width)
        new_img = res.transpose(1, 3, 0, 2).reshape(result_height, result_width, -1).sum(axis = -1)
    else:      #this implementation is custom but incredibly slow limits fps to ~1
        new_img = np.zeros((height, width))
        if padding:
            padded_frame = np.pad(frame,((kernel_height//2, kernel_height//2), (kernel_width//2, kernel_width//2)), mode = 'constant', constant_values=0 )
        else:
            padded_frame = frame
        for i in range(kernel_height//2, height-kernel_height//2):
            for j in range(kernel_width//2, width-kernel_width//2):
                kernel_area = padded_frame[i - kernel_height//2 : i+kernel_height//2+1 ,j-kernel_width//2 : j+kernel_width//2+1]  
                new_img[i,j] = np.clip(int((kernel_area * kernel).sum()), 0, 255) #very slow manual implementation

    new_img = new_img.astype(np.uint8)
    return new_img