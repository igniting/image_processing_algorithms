#include <cmath>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

int main(int argc, const char* argv[])
{
    Mat src;

    if( argc != 2 || !(src=imread(argv[1], 0)).data)
        return -1;

    Mat mask;
    Canny(src, mask, 100, 200, 3);

    Mat dst_cpu;
    cvtColor(mask, dst_cpu, CV_GRAY2BGR);
    Mat dst_gpu = dst_cpu.clone();

    GpuMat d_src(mask);
    GpuMat d_lines;
    HoughLinesBuf d_buf;
    {
        const int64 start = getTickCount();

        gpu::HoughLinesP(d_src, d_lines, d_buf, 1.0f, (float) (CV_PI / 180.0f), 50, 5);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "GPU Time : " << timeSec * 1000 << " ms" << endl;
        cout << "GPU Found : " << d_lines.cols << endl;
    }
    
    vector<Vec4i> lines_gpu;
    if (!d_lines.empty())
    {
        lines_gpu.resize(d_lines.cols);
        Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_gpu[0]);
        d_lines.download(h_lines);
    }

    for (size_t i = 0; i < lines_gpu.size(); ++i)
    {
        Vec4i l = lines_gpu[i];
        line(dst_gpu, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
    }

    imshow("source", src);
    imshow("detected lines [GPU]", dst_gpu);
    waitKey();

    return 0;
}
