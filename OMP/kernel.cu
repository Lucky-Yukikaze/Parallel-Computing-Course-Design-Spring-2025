#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <cassert>
#include <chrono>
#include <io.h>
#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void rgb2grayOpenMP(unsigned char* const d_in, unsigned char* const d_out, int imgHeight, int imgWidth)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < imgHeight; ++i)
    {
        for (int j = 0; j < imgWidth; ++j)
        {
            int idx = i * imgWidth + j;
            d_out[idx] = static_cast<unsigned char>(
                0.299f * d_in[idx * 3] +
                0.587f * d_in[idx * 3 + 1] +
                0.114f * d_in[idx * 3 + 2]);
        }
    }
}

void getFiles(string path, vector<string>& files)
{
    intptr_t hFile = 0;
    struct _finddata_t fileinfo;
    string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
    {
        do
        {
            if ((fileinfo.attrib & _A_SUBDIR))
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
            }
            else
            {
                files.push_back(p.assign(path).append("\\").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

int OpenMPFunc(string inputfilename, double& openmpsumtime)
{
    Mat srcImg = imread(inputfilename);
    if (srcImg.empty())
    {
        cerr << "无法打开图片: " << inputfilename << endl;
        return -1;
    }

    int imgHeight = srcImg.rows;
    int imgWidth = srcImg.cols;

    Mat grayImg(imgHeight, imgWidth, CV_8UC1, Scalar(0));

    // OpenMP 并行部分
    auto omp_start = chrono::high_resolution_clock::now();
    rgb2grayOpenMP(srcImg.data, grayImg.data, imgHeight, imgWidth);
    auto omp_end = chrono::high_resolution_clock::now();

    double omp_time = chrono::duration<double>(omp_end - omp_start).count();
    openmpsumtime += omp_time;
    cout << fixed << setprecision(10) << "OpenMP执行时间: " << omp_time << " s" << endl;

    // 保存灰度图
    try
    {
        string outputname = "./GrayPicture/" + inputfilename.substr(10, inputfilename.length() - 14) + "_to_gray.png";
        vector<int> compression_params = { IMWRITE_PNG_COMPRESSION, 0 };
        imwrite(outputname, grayImg, compression_params);
        cout << "保存成功 " << outputname << endl;
    }
    catch (runtime_error& ex)
    {
        cerr << "图像保存失败: " << ex.what() << endl;
        return 1;
    }

    return 0;
}

int main()
{
    string filePath = "./Picture";
    vector<string> files;
    getFiles(filePath, files);

    int size = files.size();
    cout << "图片数量：" << size << endl;

    // 用户输入线程数
    int thread_num;
    cout << "请输入使用的线程数：";
    cin >> thread_num;
    omp_set_num_threads(thread_num);

    double openmpsumtime = 0;

    for (int i = 0; i < size; i++)
    {
        cout << "处理第 " << i + 1 << " 张图片：" << files[i] << endl;
        OpenMPFunc(files[i], openmpsumtime);
        cout << endl;
    }

    cout << "总计 OpenMP 执行时间: " << openmpsumtime << " s" << endl;

    FILE* fp = fopen("time.txt", "a");
    if (fp)
    {
        fprintf(fp, "线程数：%d，OpenMP总时间：%.10lf s\n", thread_num, openmpsumtime);
        fclose(fp);
    }

    return 0;
}