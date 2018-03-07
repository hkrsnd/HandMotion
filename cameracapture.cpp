#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cv.h>
#include <vector>
using namespace std;
using namespace cv;

#define F_WIDTH 900
#define F_HEIGHT 640

VideoCapture captureVideo() {
    VideoCapture cap(0);
    // sets a property in the VideoCapture
    cap.set(CV_CAP_PROP_FRAME_WIDTH, F_WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, F_HEIGHT);
    if(!cap.isOpened()) 
    {
        printf("カメラが検出できませんでした");
        return -1;
    }
    return cap;
}

Mat getSkinImage(Mat hsv_img) {
    Mat hsv_skin_img = Mat(Size(F_WIDTH, F_HEIGHT), CV_8UC3);
    hsv_skin_img = Scalar(0,0,0); // 黒色に初期化
    // get skin part
    inRange(hsv_img, Scalar(0, 20, 20), Scalar(25, 255, 255), hsv_skin_img);
    // smoothen
    Mat structElem = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(hsv_skin_img, hsv_skin_img, MORPH_CLOSE, structElem);
    return hsv_skin_img;
}

IplImage *distanceTransform(Mat hsv_skin_img) {
    IplImage *dst_img, *dst_img_norm, *src_img;
    //距離変換 cv::Mat img(cv::Size(320, 240), CV_8UC3, cv::Scalar(0, 0, 255));
    //距離画像を計算し，表示用に結果を0-255に正規化する
    // convert Mat => IplImage*
    IplImage tmp_src_img = hsv_skin_img; 
    src_img = &tmp_src_img;
    // create dst_img after getting src_img to set same size
    dst_img = cvCreateImage (cvSize (src_img->width, src_img->height), IPL_DEPTH_32F, 1);
    dst_img_norm = cvCreateImage (cvSize (src_img->width, src_img->height), IPL_DEPTH_8U, 1);
    cvDistTransform (src_img, dst_img, CV_DIST_L2, 3, NULL, NULL);
    cvNormalize (dst_img, dst_img_norm, 0.0, 255.0, CV_MINMAX, NULL);

    return dst_img_norm;
}

vector<Point> findMaxContour(vector<vector<Point> > contours) {
    double maxarea = 0.0;
    int maxconid = 0;
    for (int i = 0; i < contours.size(); i++) {
        if (contourArea(contours[i]) > maxarea) {
            maxarea = contourArea(contours[i]);
            maxconid = i;
        }
    }
    return contours[maxconid];
}

vector<Point> findConvexHull(vector<Point> contour) {
    vector<Point> hull;
    convexHull(contour, hull, true);
    return hull;
}

vector<int> findConvexHullI(vector<Point> contour) {
    vector<int> hullI;
    convexHull(contour, hullI, true);
    return hullI;
}

// find convexity defects of each contour
vector<Vec4i> findConvexityDefects(vector<Point> contour, vector<int> hullI) {
    vector<Vec4i> condefects;

    if(hullI.size() > 3) {
        convexityDefects(contour, hullI, condefects);
    }
    return condefects;
}

// draw defecs points and line
void drawDefects(vector<Point> hull, vector<Vec4i> defects, Mat input_img) {
    // is detected flas of startpoint and endpoint
    vector<bool> isdetected_start(defects.size(), false);
    vector<bool> isdetected_end(defects.size(), false);
    int fingerNumber = 0;

    for (int i = 0; i < defects.size(); i++) {
        Vec4i v = defects[i];
        int startidx = v[0];
        Point ptStart(hull[startidx]);
        int endidx = v[1];
        Point ptEnd(hull[endidx]);
        int faridx = v[2];
        Point ptFar(hull[faridx]);
        float depth = v[3] / 256;

        if (depth > 20) {
            fingerNumber++;
            printf("%d\n", fingerNumber);
            circle(input_img, ptFar, 4,  Scalar(0,0,255));
            /*
            if (i == 0) {
                line(input_img, ptStart, ptFar, CV_RGB(0,255,0)); isdetected_start[i] = true;
                circle(input_img, ptStart, 4,  Scalar(0,0,255));
                line(input_img, ptEnd, ptFar, CV_RGB(0,255,0)); isdetected_end[i] = true;
                circle(input_img, ptEnd, 4,  Scalar(0,0,255));
                circle(input_img, ptFar, 4,  Scalar(0,0,255));
            } else if (i == defects.size()) {
                line(input_img, ptEnd, ptFar, CV_RGB(0,255,0)); isdetected_end[i] = true;
                circle(input_img, ptEnd, 4,  Scalar(0,0,255));
                circle(input_img, ptFar, 4,  Scalar(0,0,255));
            }
            else if (isdetected_end[i-1] == false) {
                line(input_img, ptStart, ptFar, CV_RGB(0,255,0)); isdetected_start[i] = true;
                circle(input_img, ptStart, 4,  Scalar(0,0,255));
            } 
            else if (isdetected_start[i+1] == false) {
                line(input_img, ptEnd, ptFar, CV_RGB(0,255,0)); isdetected_end[i] = true;
                circle(input_img, ptEnd, 4,  Scalar(0,0,255));
                circle(input_img, ptFar, 4,  Scalar(0,0,255));
            }*/
        }
    }
    printf("%d\n", fingerNumber);
}

Point findFingerPoint(vector<Point> hull, vector<Vec4i> defects) {
    int maxdepth = 0;
    int maxid = 0;
    Point maxpoint;
    for (int i = 0; i < defects.size(); i++) {
       if (defects[i][3] > maxdepth) {
           maxdepth = defects[i][3];
           maxid = defects[i][1];
           maxpoint = hull[maxid];
       }
    }
    return maxpoint;
}

void createWindows() {
    namedWindow("input_img", CV_WINDOW_AUTOSIZE);
    namedWindow("hsv_skin_img", CV_WINDOW_AUTOSIZE);
    namedWindow ("Distance Image", CV_WINDOW_AUTOSIZE);
}

void showImage(Mat input_img, Mat hsv_skin_img, IplImage *dst_img) {
    imshow("input_img", input_img);
    imshow("hsv_skin_img", hsv_skin_img);
    //imshow("Distance Image", dst_img);
    cvShowImage ("Distance Image", dst_img);
}

int main(int argc, char *argv[])
{ 
    Mat input_img;
    Mat smooth_img;
    Mat hsv_img;
    Mat hsv_skin_img;
    Mat dst_mat;
    IplImage *dst_img;

    vector<vector<Point> > contours;
    vector<Point> contour;
    vector<Vec4i> hierarchy;
    vector<Point> hull;
    vector<int> hullI;
    vector<Vec4i> defects;
    Scalar color = Scalar(255,0,0);

    VideoCapture cap = captureVideo();

    createWindows();
    while(1)
    {
        //capture video
        cap >> input_img;
        // flatten the noise
        medianBlur(input_img,smooth_img,7);
        // convert to HSV
        cvtColor(smooth_img,hsv_img,CV_BGR2HSV);
        // get skin image
        hsv_skin_img = getSkinImage(hsv_img);
        // distance transform
        dst_img = distanceTransform(hsv_skin_img);
        dst_mat = Mat(dst_img, true);
        // get contours
        findContours(dst_mat, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

        if (contours.size() > 0){
            // get Max contours
            contour = findMaxContour(contours);
            // wrap vector to pass to drawContours
            vector<vector<Point> > con(1, contour);
            // draw  maxcontour
            drawContours( input_img, con, -1, color, 1, 8);

            // find convex hull
            hull = findConvexHull(contour);
            hullI = findConvexHullI(contour);
            // wrap vector to pass to drawContours
            vector<vector<Point> > h(1, hull);
            vector<vector<int> > hI(1, hullI);

            drawContours( input_img, h, -1, color, 1, 8);
            // find Convexity defects
            defects = findConvexityDefects(contour, hullI);
            drawDefects(contour, defects, input_img);
            circle(input_img, findFingerPoint(contour, defects), 4,  Scalar(0,0,255), 2);
        }

        showImage(input_img, hsv_skin_img, dst_img);
        
        if(cv::waitKey(30) == 'q') break;
    }
}
