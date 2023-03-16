#define _USE_MATH_DEFINES

#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cmath>
#include <utility>
#include <mutex>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <viso_mono_omnidirectional.h>

#define IMG_SIZE 256



std::vector<std::pair<std::string, int>> jobs;
std::mutex jobs_mtx;

void calibrationParameters(VisualOdometryMonoOmnidirectional::parameters* pParams, double f) {

    // Taylor expansion of: r cot (r a)
    double a = M_PI * f / (180 * IMG_SIZE);
    pParams->omnidirectional_calib.length_pol = 8;
    pParams->omnidirectional_calib.pol[0] = M_PI * f / 180.0;//std::pow(a, -1.0);
    pParams->omnidirectional_calib.pol[1] = 0;
    pParams->omnidirectional_calib.pol[2] = std::pow(a, 1.0) / 3;
    pParams->omnidirectional_calib.pol[3] = 0;
    pParams->omnidirectional_calib.pol[4] = std::pow(a, 3.0) / 45;
    pParams->omnidirectional_calib.pol[5] = 0;
    pParams->omnidirectional_calib.pol[6] = std::pow(a, 5.0) / 945;
    pParams->omnidirectional_calib.pol[7] = 0;
    

    // " of theta / a
    pParams->omnidirectional_calib.length_invpol = 2;
    pParams->omnidirectional_calib.invpol[0] = 0.0;
    pParams->omnidirectional_calib.invpol[1] = 1/a;

    pParams->omnidirectional_calib.xc = IMG_SIZE / 2.0;
    pParams->omnidirectional_calib.yc = IMG_SIZE / 2.0;

    pParams->omnidirectional_calib.c = 1.0;
    pParams->omnidirectional_calib.d = 0.0;
    pParams->omnidirectional_calib.e = 0.0;

    pParams->omnidirectional_calib.height = IMG_SIZE;
    pParams->omnidirectional_calib.width = IMG_SIZE;

    // Matcher params
    pParams->match.match_radius = 50;
    pParams->match.half_resolution = 0;

    // Bucketing params
    pParams->bucket.max_features = 6;
}

// Load as grayscale 8bit
bool loadImage(std::string world, int fov, int step, cv::Mat* pImg) {
    std::stringstream ss;
    ss << "image-" << world << "-" << fov << "-" << step << ".png";

    *pImg = cv::imread(ss.str());
    if (pImg->data == nullptr) // Failed
        return false;

    pImg->convertTo(*pImg, CV_8U);
    cvtColor(*pImg, *pImg, cv::COLOR_RGB2GRAY);

    return true;
}

// Run an image through the odometer
Matrix processImage(VisualOdometryMonoOmnidirectional* pOdom, int step, cv::Mat* pImg) {
    int32_t dims[] = {pImg->size[0], pImg->size[1], pImg->size[0]};
    pOdom->process(pImg->data, dims);
    
    if (step == 0) return Matrix();

    return Matrix::inv(pOdom->getMotion());
}

void fovSeries(std::string world, int fov) {
    VisualOdometryMonoOmnidirectional::parameters params;
    calibrationParameters(&params, fov);
    VisualOdometryMonoOmnidirectional odom = VisualOdometryMonoOmnidirectional(params);

    std::stringstream ss;
    ss << "out-" << world << "-" << fov << ".txt";
    std::ofstream output(ss.str());

    cv::Mat img;
    Matrix m;
    int nMatch;
    int nInliers;
    for (int i = 0; loadImage(world, fov, i, &img); ++i) {
        m = processImage(&odom, i, &img);
        if (i == 0) continue;
        nMatch = odom.getNumberOfMatches();
        nInliers = odom.getNumberOfInliers();

        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k) {
                output << m.val[j][k] << " ";
            }
        output << std::endl;
        output << nMatch << std::endl;
        output << nInliers << std::endl;
        output << odom.ave_movement << std::endl;
    }
}

void process() {
    while(true) {
        jobs_mtx.lock();
        if (jobs.empty()) {jobs_mtx.unlock(); return;}
        std::pair<std::string, int> job = jobs.back();
        jobs.pop_back();
        jobs_mtx.unlock();

        std::cout << "Starting on " << job.first << ", " << job.second << " degrees" << std::endl;
        fovSeries(job.first, job.second);
    }
}

int main() {
    std::ifstream fovFile("fovs.txt");

    std::vector<std::string> worlds;
    int nWorlds;
    std::string buff;
    fovFile >> nWorlds;
    std::getline(fovFile, buff);
    for (int i = 0; i < nWorlds; ++i) {
        std::getline(fovFile, buff);
        buff.erase(buff.find_last_not_of(" \n\t\r") + 1);
        worlds.push_back(buff);
    }

    std::vector<int> fovs;
    int fov;
    while (fovFile >> fov) fovs.push_back(fov);

    // Process separately each fov with data gathered
    for (std::string world : worlds)
        for (int fov : fovs) {
            jobs.push_back(std::make_pair(world, fov));
        }

    std::vector<std::thread> workers;
    for (int i = 0; i < 1; ++i) {
        std::thread t(process);
        workers.push_back(std::move(t));
    }

    for (int i = 0; i < workers.size(); ++i) {
        workers[i].join();
    }
    
    std::cout << "Done!" << std::endl;
}