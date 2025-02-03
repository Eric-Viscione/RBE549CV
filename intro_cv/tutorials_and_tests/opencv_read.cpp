//g++ -o opencv_read opencv_read.cpp $(pkg-config --cflags --libs opencv4)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;

int main()
{
    std::string image_path = samples::findFilcde("starry_night.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not reaed the image" << image_path << std::endl;
        return 1;
    }
    imshow("Display window", img);
    int k = waitKey(0);
    if(k == 's')
    {
        imwrite("starry_night.png", img);
    }
    return 0;
}