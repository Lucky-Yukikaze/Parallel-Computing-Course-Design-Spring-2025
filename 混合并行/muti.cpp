#define _CRT_SECURE_NO_WARNINGS

#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <omp.h>

namespace fs = std::filesystem;

// OpenMP优化的灰度转换函数
void rgb2grayOpenMP(unsigned char* const d_in, unsigned char* const d_out, int imgHeight, int imgWidth) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < imgHeight; ++i) {
        for (int j = 0; j < imgWidth; ++j) {
            int idx = i * imgWidth + j;
            d_out[idx] = static_cast<unsigned char>(
                0.299f * d_in[idx * 3] +
                0.587f * d_in[idx * 3 + 1] +
                0.114f * d_in[idx * 3 + 2]);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 设置输入和输出文件夹路径
    const std::string inputFolderPath = "D:/muti/x64/Debug/Picture";
    const std::string outputFolderPath = "D:/muti/x64/Debug/GrayPicture";

    // 检查输入路径
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
    int totalImages = 0;
    if (rank == 0) totalImages = imagePaths.size();
    MPI_Bcast(&totalImages, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 广播路径数据
    const int MAX_PATH_LEN = 260;
    char* allPaths = new char[totalImages * MAX_PATH_LEN]();
    if (rank == 0) {
        for (int i = 0; i < totalImages; ++i) {
            strncpy(&allPaths[i * MAX_PATH_LEN], imagePaths[i].c_str(), MAX_PATH_LEN - 1);
        }
    }
    MPI_Bcast(allPaths, totalImages * MAX_PATH_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    // 任务分配
    int chunk = totalImages / size;
    int remainder = totalImages % size;
    int start = rank * chunk + std::min(rank, remainder);
    int end = start + chunk + (rank < remainder ? 1 : 0);
    end = std::min(end, totalImages);

    // 提取本地任务
    std::vector<std::string> localPaths;
    for (int i = start; i < end; ++i) {
        localPaths.emplace_back(&allPaths[i * MAX_PATH_LEN]);
    }
    delete[] allPaths;

    // 创建输出目录
    if (rank == 0 && !fs::exists(outputFolderPath)) {
        fs::create_directories(outputFolderPath);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 设置OpenMP线程数（可通过命令行参数控制）
    int omp_threads = 2;  // 默认2线程
    if (argc > 1) omp_threads = atoi(argv[1]);
    omp_set_num_threads(omp_threads);

    // 同步计时
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // 处理本地图片（MPI进程间并行 + OpenMP线程级并行）
    for (const auto& path : localPaths) {
        cv::Mat srcImg = cv::imread(path);
        if (srcImg.empty()) {
            std::cerr << "进程 " << rank << ": 读取失败: " << path << std::endl;
            continue;
        }

        // 准备灰度图像
        cv::Mat grayImg(srcImg.rows, srcImg.cols, CV_8UC1);

        // 使用混合并行处理
        rgb2grayOpenMP(srcImg.data, grayImg.data, srcImg.rows, srcImg.cols);

        // 保存结果
        fs::path p(path);
        std::string outputName = outputFolderPath + "/hybrid_rank" +
            std::to_string(rank) + "_" + p.filename().string();

        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);  // PNG压缩参数
        compression_params.push_back(0);

        // 保存图像时传入压缩参数
        if (!cv::imwrite(outputName, grayImg, compression_params)) {
            std::cerr << "进程 " << rank << ": 保存失败: " << outputName << std::endl;
        }
    }

    // 计算总时间
    MPI_Barrier(MPI_COMM_WORLD);
    double totalTime = MPI_Wtime() - startTime;

    // 汇总统计信息
    if (rank == 0) {
        std::cout
            << "总图片数: " << totalImages << "\n"
            << "MPI进程数: " << size << "\n"
            << "OpenMPOpenMP线程/进程: " << omp_threads << "\n"
            << "总耗时: " << totalTime << " 秒\n";
    }

    MPI_Finalize();
    return 0;
}