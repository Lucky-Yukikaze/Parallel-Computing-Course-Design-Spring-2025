#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <chrono>
#include <io.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 灰度转换（串行）
void rgb2graySerial(const unsigned char* const in, unsigned char* const out, int height, int width)
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            int idx = i * width + j;
            out[idx] = 0.299f * in[idx * 3] + 0.587f * in[idx * 3 + 1] + 0.114f * in[idx * 3 + 2];
        }
    }
}

// 递归获取文件
void getFiles(string path, vector<string>& files)
{
    intptr_t hFile = 0;
    struct _finddata_t fileinfo;
    string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
    {
        do
        {
            if (!(fileinfo.attrib & _A_SUBDIR))
                files.push_back(p.assign(path).append("\\").append(fileinfo.name));
            else if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

// 主处理函数（串行）
int SerialFunc(string inputfilename, double& cpusumtime)
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
    int hist[256] = { 0 };

    auto cpu_start = chrono::high_resolution_clock::now();

    rgb2graySerial(srcImg.data, grayImg.data, imgHeight, imgWidth);

    auto cpu_end = chrono::high_resolution_clock::now();
    auto cpu_duration = chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start);
    double cpu_time = cpu_duration.count() / 1e6;
    cpusumtime += cpu_time;

    cout << fixed << setprecision(10) << "CPU 执行时间: " << cpu_time << " s" << endl;

    // 保存灰度图
    try
    {
        string outputname = "./GrayPicture/" + inputfilename.substr(10, inputfilename.length() - 14) + "_to_gray.png";
        vector<int> compression_params = { IMWRITE_PNG_COMPRESSION, 0 };
        imwrite(outputname, grayImg, compression_params);
        cout << "保存成功: " << outputname << endl;
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
    cout << "共检测到图片：" << size << " 张" << endl;

    double cpusumtime = 0;

    for (int i = 0; i < size; ++i)
    {
        cout << "\n正在处理第 " << i + 1 << " 张图片：" << files[i] << endl;
        SerialFunc(files[i], cpusumtime);
    }

    cout << "\nCPU 总处理时间: " << cpusumtime << " s" << endl;

    FILE* fp = fopen("time.txt", "a");
    if (fp)
    {
        fprintf(fp, "CPU总时间： %.10lf s\n", cpusumtime);
        fclose(fp);
    }

    return 0;
}
