#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

class ImageBuffer {
private:
  std::queue<cv::Mat> images;
  std::string folder_path;
  int current_index = 0;
  int buffer_size = 10;
  int prefetch_threshold = 5;
  std::thread loader_thread;
  std::mutex mutex;

  void loader() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);
      if (images.size() < prefetch_threshold) {
        for (int i = 0; i < prefetch_threshold; ++i) {
          std::string image_path =
              folder_path + "/" + std::to_string(current_index + i) + ".jpg";
          cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
          images.push(image);
        }
        current_index += prefetch_threshold;
      }
      lock.unlock();
    }
  }

public:
  ImageBuffer(const std::string &folder_path) : folder_path(folder_path) {
    loader_thread = std::thread(&ImageBuffer::loader, this);
  }

  cv::Mat get_image() {
    std::unique_lock<std::mutex> lock(mutex);
    if (!images.empty()) {
      cv::Mat image = images.front();
      images.pop();
      return image;
    } else {
      return cv::Mat::zeros(480, 640, CV_8UC1);
    }
  }
};

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(image_buffer, m) {
  py::class_<ImageBuffer>(m, "ImageBuffer")
      .def(py::init<const std::string &>())
      .def("get_image", &ImageBuffer::get_image);
}
