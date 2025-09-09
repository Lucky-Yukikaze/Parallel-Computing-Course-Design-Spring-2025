#define _CRT_SECURE_NO_WARNINGS

#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// 手动实现灰度转换函数
cv::Mat manualBGR2Gray(const cv::Mat& img) {
    CV_Assert(img.type() == CV_8UC3);

    cv::Mat gray(img.rows, img.cols, CV_8UC1);

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
            float grayValue = 0.299f * pixel[2] + 
                0.587f * pixel[1] +  
                0.114f * pixel[0];  
            gray.at<uchar>(y, x) = static_cast<uchar>(grayValue);
        }
    }

    return gray;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 设置输入和输出文件夹路径
    std::string inputFolderPath = "D:/MPI/Picture";
    std::string outputFolderPath = "D:/MPI/GrayPicture";

    // 检查输入路径是否存在
    if (!fs::exists(inputFolderPath)) {
        std::cerr << "进程 " << rank << ": 错误 - 文件夹未找到: " << inputFolderPath << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 主进程收集图像路径
    std::vector<std::string> imagePaths;
    if (rank == 0) {
        for (const auto& entry : fs::directory_iterator(inputFolderPath)) {
            if (entry.is_regular_file()) {
                imagePaths.push_back(entry.path().string());
            }
        }
        std::cout << "找到总图片数: " << imagePaths.size() << std::endl;
    }

    // 广播图像总数
    int totalImages = imagePaths.size();
    MPI_Bcast(&totalImages, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 广播图像路径
    const int MAX_PATH_LEN = 260;
    char* allPaths = new char[totalImages * MAX_PATH_LEN];
    if (rank == 0) {
        for (int i = 0; i < totalImages; ++i) {
            strncpy(&allPaths[i * MAX_PATH_LEN], imagePaths[i].c_str(), MAX_PATH_LEN);
        }
    }
    MPI_Bcast(allPaths, totalImages * MAX_PATH_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    //块分配逻辑
    int chunk_size = totalImages / size;
    int remainder = totalImages % size;

    int start, end;
    if (rank < remainder) {
        start = rank * (chunk_size + 1);
        end = start + chunk_size + 1;
    }
    else {
        start = rank * chunk_size + remainder;
        end = start + chunk_size;
    }
    end = std::min(end, totalImages);

    // 提取本进程分配到的路径
    std::vector<std::string> localPaths;
    for (int i = start; i < end; ++i) {
        localPaths.emplace_back(&allPaths[i * MAX_PATH_LEN]);
    }
    delete[] allPaths;

    std::cout << "进程 " << rank << ": 分配到 " << localPaths.size()
        << " 张图片 (块范围: " << start << "-" << end - 1 << ")" << std::endl;

    // 确保输出目录存在
    if (!fs::exists(outputFolderPath)) {
        fs::create_directories(outputFolderPath);
    }

    // 同步所有进程并开始计时
    MPI_Barrier(MPI_COMM_WORLD);
    double startWallTime = MPI_Wtime();

    // 图像处理
    for (int imgIdx = 0; imgIdx < localPaths.size(); ++imgIdx) {
        const auto& path = localPaths[imgIdx];
        std::cout << "进程 " << rank << ": 正在处理图片 " << imgIdx + 1
            << " / " << localPaths.size() << ": " << path << std::endl;

        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "进程 " << rank << ": 读取图片失败: " << path << std::endl;
            continue;
        }

        // 使用手动实现的灰度转换
        cv::Mat gray = manualBGR2Gray(img);

        fs::path p(path);
        std::string outputName = "processed_rank" + std::to_string(rank) + "_" + p.stem().string() + ".png";
        std::string outputPath = outputFolderPath + "/" + outputName;

        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(0);  // 设置为无压缩

        if (!cv::imwrite(outputPath, gray, compression_params)) {
            std::cerr << "进程 " << rank << ": 保存图片失败: " << outputName << std::endl;
        }
    }

    // 同步后结束计时
    MPI_Barrier(MPI_COMM_WORLD);
    double endWallTime = MPI_Wtime();
    double localElapsed = endWallTime - startWallTime;

    // 汇总最大时间（实际运行时间）
    double globalElapsed;
    MPI_Reduce(&localElapsed, &globalElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 主进程输出结果
    if (rank == 0) {
        std::cout << "配置信息:" << std::endl;
        std::cout << " - 总图片数: " << totalImages << std::endl;
        std::cout << " - 进程数: " << size << std::endl;
        std::cout << " - 实际运行时间: " << globalElapsed << " 秒" << std::endl;
    }

    MPI_Finalize();
    return 0;
}