
#ifndef IMAGE_DATA_HPP
#define IMAGE_DATA_HPP


#include "opencv2/core.hpp"
#include "boost/shared_ptr.hpp"


struct ImgData {
    double timeStampToSec;
    cv::Mat image;
};

typedef boost::shared_ptr<ImgData> ImageDataPtr;


#endif // IMAGE_DATA_HPP