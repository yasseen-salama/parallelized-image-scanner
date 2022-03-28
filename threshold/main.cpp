#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>

#include <iostream>
using namespace cv;

using namespace std;

bool testEnv = false;

void thresholdIntegral(cv::Mat& inputMat, cv::Mat& outputMat)
{
    // accept only char type matrices
    CV_Assert(!inputMat.empty());
    CV_Assert(inputMat.depth() == CV_8U);
    CV_Assert(inputMat.channels() == 1);
    CV_Assert(!outputMat.empty());
    CV_Assert(outputMat.depth() == CV_8U);
    CV_Assert(outputMat.channels() == 1);

    auto start_time = omp_get_wtime();

    // create the integral image
    cv::Mat sumMat;
    cv::integral(inputMat, sumMat);

    CV_Assert(sumMat.depth() == CV_32S);
    CV_Assert(sizeof(int) == 4);

    // rows -> height -> y
    int nRows = inputMat.rows;
    // cols -> width -> x
    int nCols = inputMat.cols;
    int S = MAX(nRows, nCols) / 32;
    double T = 0.15;

    // CV_Assert(sizeof(int) == 4);
    int* p_y1, * p_y2;
    uchar* p_inputMat, * p_outputMat;

    // perform thresholding
    int s2 = S / 2;
    int x1, y1, x2, y2, count, sum;

    for (int i = 0; i < nRows; ++i)
    {

        y1 = i - s2;
        y2 = i + s2;

        if (y1 < 0)
        {
            y1 = 0;
        }
        if (y2 >= nRows)
        {
            y2 = nRows - 1;
        }

        /*    y1++;
            y2++;*/

        p_y1 = sumMat.ptr<int>(y1);
        p_y2 = sumMat.ptr<int>(y2);
        p_inputMat = inputMat.ptr<uchar>(i);
        p_outputMat = outputMat.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j)
        {
            // set the SxS region
            x1 = j - s2;
            x2 = j + s2;

            if (x1 < 0)
            {
                x1 = 0;
            }
            if (x2 >= nCols)
            {
                x2 = nCols - 1;
            }

            /* x1++;
             x2++;*/

            count = (x2 - x1) * (y2 - y1);

            // I(x,y)=s(x2,y2)-s(x1,y2)-s(x2,y1)+s(x1,x1)
            sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];

            if ((int)(p_inputMat[j] * count) < (int)(sum * (1.0 - T)))
                p_outputMat[j] = 0;
            else
                p_outputMat[j] = 255;


        }
    }
    auto run_time = omp_get_wtime() - start_time;
    cout << "\n" << "************" << "run_time: " << run_time << " S " << "***********" << "\n" << endl;

}
void thresholdIntegralParallel(cv::Mat& inputMat, cv::Mat& outputMat)
{

    // accept only char type matrices
    CV_Assert(!inputMat.empty());
    CV_Assert(inputMat.depth() == CV_8U);
    CV_Assert(inputMat.channels() == 1);
    CV_Assert(!outputMat.empty());
    CV_Assert(outputMat.depth() == CV_8U);
    CV_Assert(outputMat.channels() == 1);

    auto start_time_p = omp_get_wtime();

    // create the integral image
    cv::Mat sumMat;
    cv::integral(inputMat, sumMat);

    CV_Assert(sumMat.depth() == CV_32S);
    CV_Assert(sizeof(int) == 4);

    // rows -> height -> y
    int nRows = inputMat.rows;
    // cols -> width -> x
    int nCols = inputMat.cols;
    int S = MAX(nRows, nCols) / 32;
    double T = 0.15;

    // CV_Assert(sizeof(int) == 4);


        // perform thresholding
    int s2 = S / 2;

#pragma omp parallel for  num_threads(2) shared (nRows,nCols,s2,T,sumMat)



    // first for loop is to selcet the rows for pointer and mask  

    for (int i = 0; i < nRows; ++i)
    {
        int x1, y1, x2, y2, count, sum;
        int* p_y1, * p_y2;
        uchar* p_inputMat, * p_outputMat;

        y1 = i - s2;
        y2 = i + s2;

        if (y1 < 0)
        {
            y1 = 0;
        }
        if (y2 >= nRows)
        {
            y2 = nRows - 1;
        }

        /*    y1++;
            y2++;*/

        p_y1 = sumMat.ptr<int>(y1); // pointer of mask
        p_y2 = sumMat.ptr<int>(y2); // pointer of mask
        p_inputMat = inputMat.ptr<uchar>(i);
        p_outputMat = outputMat.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j)
        {
            // set the SxS region
            x1 = j - s2;
            x2 = j + s2;

            if (x1 < 0)
            {
                x1 = 0;
            }
            if (x2 >= nCols)
            {
                x2 = nCols - 1;
            }

            /* x1++;
             x2++;*/

            count = (x2 - x1) * (y2 - y1);

            // I(x,y)=s(x2,y2)-s(x1,y2)-s(x2,y1)+s(x1,x1)
            sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];

            if ((int)(p_inputMat[j] * count) < (int)(sum * (1.0 - T)))
                p_outputMat[j] = 0;
            else
                p_outputMat[j] = 255;

        }
    }

    auto run_time_p = omp_get_wtime() - start_time_p;
    cout << "\n" << "************" << "run_time_Parallel: " << run_time_p << " S " << "***********" << "\n" << endl;
}

int main(int argc, char* argv[])
{

    // Load the image

    const char* default_file = "bookpage.jpg";
    const char* filename = argc >= 2 ? argv[1] : default_file;


    // Loads an image
    cv::Mat src = cv::imread(samples::findFile(filename), cv::ImreadModes::IMREAD_GRAYSCALE);
    cv::Mat mSrc;
    //  Mat image_blurred_with_3x3_kernel;
     // GaussianBlur(src, image_blurred_with_3x3_kernel, Size(3, 3), 0);
      //src = image_blurred_with_3x3_kernel;

      // Check if image is loaded fine
    if (src.empty())
    {
        cerr << "Problem loading image!!!" << endl;
        return -1;
    }

    // Show source image
    try
    {
        cv::imshow("src", src);
    }
    catch (const cv::Exception& e)
    {
        cout << "Are we testEnv?" << endl;
        testEnv = true;
    }
    //! [load_image]

    //! [gray]
    // Transform source image to gray
    cv::Mat gray;

    if (src.channels() == 3)
    {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

        // Show gray image
        cv::imshow("gray", gray);
    }
    else
    {
        gray = src;
    }

    cout << "TEST" << endl;

    //! [gray]

    ////! [bin_1]
    //cv::Mat bw1;
    //cv::adaptiveThreshold(gray, bw1, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, -2);

    //// Show binary image
    //if (testEnv)
    //{
    //    cv::imwrite("threshold_opencv.png", bw1);
    //}
    //else
    //{
    //    cv::imshow("threshold_opencv", bw1);
    //}
    ////! [bin_1]


    // [bin_Parallel]
    cv::Mat bw1 = cv::Mat::zeros(gray.size(), CV_8UC1);
    thresholdIntegralParallel(gray, bw1);

    // Show binary image
    if (testEnv)
    {
        cv::imwrite("threshold_integral_Parallel.png", bw1);
    }
    else
    {
        cv::imshow("threshold_integral_Parallel", bw1);
    }
    //! [bin_Parallel]
    //
    // [bin]
    cv::Mat bw2 = cv::Mat::zeros(gray.size(), CV_8UC1);
    thresholdIntegral(gray, bw2);

    // Show binary image
    if (testEnv)
    {
        cv::imwrite("threshold_integral.png", bw2);
    }
    else
    {
        cv::imshow("threshold_integral", bw2);
    }
    //! [bin]

    cv::waitKey(0);
    return 0;
}