#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cv.h>
#include <vector>
using namespace std;
using namespace cv;

#define F_WIDTH 460
#define F_HEIGHT 640

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

int main(int argc, char *argv[])
{
    cv::VideoCapture cap(0);
    // sets a property in the VideoCapture
    cap.set(CV_CAP_PROP_FRAME_WIDTH, F_WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, F_HEIGHT);
    if(!cap.isOpened()) 
    {
        printf("カメラが検出できませんでした");
        return -1;
    }
    cv::Mat input_img;
    //cv::Mat hsv_skin_img= cv::Mat(cv::Size(320,240),CV_8UC3); // 符号なし8ビット整数型，3チャンネル
    cv::Mat hsv_skin_img = Mat(Size(F_WIDTH, F_HEIGHT), CV_8UC3);
    cv::Mat smooth_img;
    cv::Mat hsv_img;
    //cv::Mat dst_img;
    //cv::Mat dst_img_norm;
    // (2)処理結果の距離画像出力用の画像領域と表示ウィンドウを確保
    IplImage *dst_img, *dst_img_norm, *src_img;
    //dst_img = cvCreateImage (cvSize (src_img->width, src_img->height), IPL_DEPTH_32F, 1);
    //dst_img_norm = cvCreateImage (cvSize (F_WIDTH, F_HEIGHT), IPL_DEPTH_8U, 1);

    //cv::Mat frame(Size(F_WIDTH, F_HEIGHT), IPL_DEPTH_8U);
    std::vector<vector<Point> > contours;
    std::vector<Point> contour;
    std::vector<Vec4i> hierarchy;
    
    cv::namedWindow("input_img", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("hsv_skin_img", CV_WINDOW_AUTOSIZE);
    //cv::namedWindow("contours_img", CV_WINDOW_AUTOSIZE);

    while(1)
    {
        hsv_skin_img = cv::Scalar(0,0,0); // 黒色に初期化
        cap >> input_img;
        cv::medianBlur(input_img,smooth_img,7); //ノイズがあるので平滑化
        cv::cvtColor(smooth_img,hsv_img,CV_BGR2HSV);    //HSVに変換     (stc, output, int)

        // 肌色の抜き出し
        inRange(hsv_img, Scalar(0, 20, 20), Scalar(25, 255, 255), hsv_skin_img);
        // 抜き出した部分をなめらかにする処理
        //Mat structElem = getStructuringElement(MORPH_RECT, Size(3, 3));
        //morphologyEx(hsv_skin_img, hsv_skin_img, MORPH_CLOSE, structElem);

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
        
        Mat gray_dst_mat(dst_img_norm, true);

        //cvtColor(gray_dst_mat, gray_dst_mat, CV_RGB2GRAY);
        // get countous
        findContours(gray_dst_mat, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

        // getMaxAreaContour
        if (contours.size() > 0){
        contour = findMaxContour(contours);

            Scalar color(0,0,255);
            vector<vector<Point> > con(1);
            con[0] = contour;
            drawContours( input_img, con, -1, color, 1, 8);
        }
        //findContours( gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        //imshow( "input_img", dst );
        cv::imshow("input_img",input_img);
        cv::imshow("hsv_skin_img",hsv_skin_img);
        cvNamedWindow ("Source", CV_WINDOW_AUTOSIZE);
        cvNamedWindow ("Distance Image", CV_WINDOW_AUTOSIZE);
        cvShowImage ("Source", src_img);
        cvShowImage ("Distance Image", dst_img_norm);
        if(cv::waitKey(30) == 'q')
        {
            break;
        }
    }
}
