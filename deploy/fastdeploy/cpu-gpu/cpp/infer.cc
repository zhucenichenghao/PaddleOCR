// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <vector>

#include "curl/curl.h"  // has to go before opencv headers
using namespace std;

#include "crow.h"
#include "fastdeploy/vision.h"
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

// curl writefunction to be passed as a parameter
//  we can't ever expect to get the whole image in one piece,
//  every router / hub is entitled to fragment it into parts
//  (like 1-8k at a time),
//  so insert the part at the end of our stream.
size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata) {
  vector<uchar> *stream = (vector<uchar> *)userdata;
  size_t count = size * nmemb;
  stream->insert(stream->end(), ptr, ptr + count);
  return count;
}

// function to retrieve the image as cv::Mat data type
cv::Mat curlImg(const char *img_url, int timeout = 10) {
  vector<uchar> stream;
  CURL *curl = curl_easy_init();
  curl_easy_setopt(curl, CURLOPT_URL, img_url);  // the img url
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
                   write_data);  // pass the writefunction
  curl_easy_setopt(curl, CURLOPT_WRITEDATA,
                   &stream);  // pass the stream ptr to the writefunction
  curl_easy_setopt(curl, CURLOPT_TIMEOUT,
                   timeout);               // timeout if curl_easy hangs,
  CURLcode res = curl_easy_perform(curl);  // start curl
  curl_easy_cleanup(curl);                 // cleanup
  return cv::imdecode(stream, -1);         // 'keep-as-is'
}

void InitAndInfer(const std::string &det_model_dir,
                  const std::string &cls_model_dir,
                  const std::string &rec_model_dir,
                  const std::string &rec_label_file,
                  const std::string &image_file,
                  const fastdeploy::RuntimeOption &option) {
  auto det_model_file = det_model_dir + sep + "inference.pdmodel";
  auto det_params_file = det_model_dir + sep + "inference.pdiparams";

  auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
  auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";

  auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
  auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";

  auto det_option = option;
  auto cls_option = option;
  auto rec_option = option;

  // The cls and rec model can inference a batch of images now.
  // User could initialize the inference batch size and set them after create
  // PP-OCR model.
  int cls_batch_size = 1;
  int rec_batch_size = 6;

  // If use TRT backend, the dynamic shape will be set as follow.
  // We recommend that users set the length and height of the detection model to
  // a multiple of 32.
  // We also recommend that users set the Trt input shape as follow.
  det_option.SetTrtInputShape("x", {1, 3, 64, 64}, {1, 3, 640, 640},
                              {1, 3, 960, 960});
  cls_option.SetTrtInputShape("x", {1, 3, 48, 10}, {cls_batch_size, 3, 48, 320},
                              {cls_batch_size, 3, 48, 1024});
  rec_option.SetTrtInputShape("x", {1, 3, 48, 10}, {rec_batch_size, 3, 48, 320},
                              {rec_batch_size, 3, 48, 2304});

  // Users could save TRT cache file to disk as follow.
  // det_option.SetTrtCacheFile(det_model_dir + sep + "det_trt_cache.trt");
  // cls_option.SetTrtCacheFile(cls_model_dir + sep + "cls_trt_cache.trt");
  // rec_option.SetTrtCacheFile(rec_model_dir + sep + "rec_trt_cache.trt");

  auto det_model = fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, det_option);
  auto cls_model = fastdeploy::vision::ocr::Classifier(
      cls_model_file, cls_params_file, cls_option);
  auto rec_model = fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, rec_label_file, rec_option);

  assert(det_model.Initialized());
  assert(cls_model.Initialized());
  assert(rec_model.Initialized());

  // Parameters settings for pre and post processing of Det/Cls/Rec Models.
  // All parameters are set to default values.
  det_model.GetPreprocessor().SetMaxSideLen(960);
  det_model.GetPostprocessor().SetDetDBThresh(0.3);
  det_model.GetPostprocessor().SetDetDBBoxThresh(0.6);
  det_model.GetPostprocessor().SetDetDBUnclipRatio(1.5);
  det_model.GetPostprocessor().SetDetDBScoreMode("slow");
  det_model.GetPostprocessor().SetUseDilation(0);
  cls_model.GetPostprocessor().SetClsThresh(0.9);

  // The classification model is optional, so the PP-OCR can also be connected
  // in series as follows
  // auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &rec_model);
  auto ppocr_v3 =
      fastdeploy::pipeline::PPOCRv3(&det_model, &cls_model, &rec_model);

  // Set inference batch size for cls model and rec model, the value could be -1
  // and 1 to positive infinity.
  // When inference batch size is set to -1, it means that the inference batch
  // size
  // of the cls and rec models will be the same as the number of boxes detected
  // by the det model.
  ppocr_v3.SetClsBatchSize(cls_batch_size);
  ppocr_v3.SetRecBatchSize(rec_batch_size);

  if (!ppocr_v3.Initialized()) {
    std::cerr << "Failed to initialize PP-OCR." << std::endl;
    return;
  }

  auto im = curlImg(image_file.c_str());
  // auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::OCRResult result;
  if (!ppocr_v3.Predict(&im, &result)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << result.Str() << std::endl;

  auto vis_im = fastdeploy::vision::VisOcr(im_bak, result);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

fastdeploy::pipeline::PPOCRv3 Init(const std::string &det_model_dir,
                                   const std::string &cls_model_dir,
                                   const std::string &rec_model_dir,
                                   const std::string &rec_label_file,
                                   const fastdeploy::RuntimeOption &option) {
  auto det_model_file = det_model_dir + sep + "inference.pdmodel";
  auto det_params_file = det_model_dir + sep + "inference.pdiparams";

  auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
  auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";

  auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
  auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";

  auto det_option = option;
  auto cls_option = option;
  auto rec_option = option;

  // The cls and rec model can inference a batch of images now.
  // User could initialize the inference batch size and set them after create
  // PP-OCR model.
  int cls_batch_size = 1;
  int rec_batch_size = 6;

  // If use TRT backend, the dynamic shape will be set as follow.
  // We recommend that users set the length and height of the detection model to
  // a multiple of 32.
  // We also recommend that users set the Trt input shape as follow.
  det_option.SetTrtInputShape("x", {1, 3, 64, 64}, {1, 3, 640, 640},
                              {1, 3, 960, 960});
  cls_option.SetTrtInputShape("x", {1, 3, 48, 10}, {cls_batch_size, 3, 48, 320},
                              {cls_batch_size, 3, 48, 1024});
  rec_option.SetTrtInputShape("x", {1, 3, 48, 10}, {rec_batch_size, 3, 48, 320},
                              {rec_batch_size, 3, 48, 2304});

  // Users could save TRT cache file to disk as follow.
  // det_option.SetTrtCacheFile(det_model_dir + sep + "det_trt_cache.trt");
  // cls_option.SetTrtCacheFile(cls_model_dir + sep + "cls_trt_cache.trt");
  // rec_option.SetTrtCacheFile(rec_model_dir + sep + "rec_trt_cache.trt");

  auto det_model = fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, det_option);
  auto cls_model = fastdeploy::vision::ocr::Classifier(
      cls_model_file, cls_params_file, cls_option);
  auto rec_model = fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, rec_label_file, rec_option);

  assert(det_model.Initialized());
  assert(cls_model.Initialized());
  assert(rec_model.Initialized());

  // Parameters settings for pre and post processing of Det/Cls/Rec Models.
  // All parameters are set to default values.
  det_model.GetPreprocessor().SetMaxSideLen(960);
  det_model.GetPostprocessor().SetDetDBThresh(0.3);
  det_model.GetPostprocessor().SetDetDBBoxThresh(0.6);
  det_model.GetPostprocessor().SetDetDBUnclipRatio(1.5);
  det_model.GetPostprocessor().SetDetDBScoreMode("slow");
  det_model.GetPostprocessor().SetUseDilation(0);
  cls_model.GetPostprocessor().SetClsThresh(0.9);

  // The classification model is optional, so the PP-OCR can also be connected
  // in series as follows
  // auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &rec_model);
  auto ppocr_v3 =
      fastdeploy::pipeline::PPOCRv3(&det_model, &cls_model, &rec_model);

  // Set inference batch size for cls model and rec model, the value could be -1
  // and 1 to positive infinity.
  // When inference batch size is set to -1, it means that the inference batch
  // size
  // of the cls and rec models will be the same as the number of boxes detected
  // by the det model.
  ppocr_v3.SetClsBatchSize(cls_batch_size);
  ppocr_v3.SetRecBatchSize(rec_batch_size);

  if (!ppocr_v3.Initialized()) {
    std::cerr << "Failed to initialize PP-OCR." << std::endl;
  }
  return ppocr_v3;
}

std::vector<std::string> Infer(fastdeploy::pipeline::PPOCRv3 &ppocr_v3,
                               const std::string &image_file) {
  auto im = curlImg(image_file.c_str());
  if (im.empty()) {
    return {};
  }

  // auto im = cv::imread(image_file);
  fastdeploy::vision::OCRResult result;
  if (!ppocr_v3.Predict(&im, &result)) {
    std::cerr << "Failed to predict." << std::endl;
    return {};
  }
  return result.text;
}

void build(const std::string &det_model_dir, const std::string &cls_model_dir,
           const std::string &rec_model_dir, const std::string &rec_label_file,
           const fastdeploy::RuntimeOption &option) {
  int fd = socket(PF_INET, SOCK_STREAM, 0);
  if (fd == -1) {
    printf("create socket error %d\n", errno);
    exit(1);
  }

  int on = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
  struct sockaddr_in addr;
  addr.sin_family = PF_INET;
  addr.sin_port = htons(8081);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  auto det_model_file = det_model_dir + sep + "inference.pdmodel";
  auto det_params_file = det_model_dir + sep + "inference.pdiparams";

  auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
  auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";

  auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
  auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";

  auto det_option = option;
  auto cls_option = option;
  auto rec_option = option;

  // The cls and rec model can inference a batch of images now.
  // User could initialize the inference batch size and set them after create
  // PP-OCR model.
  int cls_batch_size = 1;
  int rec_batch_size = 6;

  // If use TRT backend, the dynamic shape will be set as follow.
  // We recommend that users set the length and height of the detection model to
  // a multiple of 32.
  // We also recommend that users set the Trt input shape as follow.
  det_option.SetTrtInputShape("x", {1, 3, 64, 64}, {1, 3, 640, 640},
                              {1, 3, 960, 960});
  cls_option.SetTrtInputShape("x", {1, 3, 48, 10}, {cls_batch_size, 3, 48, 320},
                              {cls_batch_size, 3, 48, 1024});
  rec_option.SetTrtInputShape("x", {1, 3, 48, 10}, {rec_batch_size, 3, 48, 320},
                              {rec_batch_size, 3, 48, 2304});

  // Users could save TRT cache file to disk as follow.
  // det_option.SetTrtCacheFile(det_model_dir + sep + "det_trt_cache.trt");
  // cls_option.SetTrtCacheFile(cls_model_dir + sep + "cls_trt_cache.trt");
  // rec_option.SetTrtCacheFile(rec_model_dir + sep + "rec_trt_cache.trt");

  auto det_model = fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, det_option);
  auto cls_model = fastdeploy::vision::ocr::Classifier(
      cls_model_file, cls_params_file, cls_option);
  auto rec_model = fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, rec_label_file, rec_option);

  assert(det_model.Initialized());
  assert(cls_model.Initialized());
  assert(rec_model.Initialized());

  // Parameters settings for pre and post processing of Det/Cls/Rec Models.
  // All parameters are set to default values.
  det_model.GetPreprocessor().SetMaxSideLen(960);
  det_model.GetPostprocessor().SetDetDBThresh(0.3);
  det_model.GetPostprocessor().SetDetDBBoxThresh(0.6);
  det_model.GetPostprocessor().SetDetDBUnclipRatio(1.5);
  det_model.GetPostprocessor().SetDetDBScoreMode("slow");
  det_model.GetPostprocessor().SetUseDilation(0);
  cls_model.GetPostprocessor().SetClsThresh(0.9);

  // The classification model is optional, so the PP-OCR can also be connected
  // in series as follows
  // auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &rec_model);
  auto ppocr_v3 =
      fastdeploy::pipeline::PPOCRv3(&det_model, &cls_model, &rec_model);

  // Set inference batch size for cls model and rec model, the value could be -1
  // and 1 to positive infinity.
  // When inference batch size is set to -1, it means that the inference batch
  // size
  // of the cls and rec models will be the same as the number of boxes detected
  // by the det model.
  ppocr_v3.SetClsBatchSize(cls_batch_size);
  ppocr_v3.SetRecBatchSize(rec_batch_size);

  if (!ppocr_v3.Initialized()) {
    std::cerr << "Failed to initialize PP-OCR." << std::endl;
  }
  crow::SimpleApp app;
  app.multithreaded();
  CROW_ROUTE(app, "/detect")
      .methods("POST"_method)([&ppocr_v3](const crow::request &req) {
        auto x = crow::json::load(req.body);
        if (!x) return crow::response(400);
        auto link = x["link"].s();
        auto result = Infer(ppocr_v3, link);
        crow::json::wvalue y;
        y["texts"] = result;
        y["ret"] = 1;
        auto resp = crow::response(y.dump());
        resp.add_header("Content-Type", "application/json");
        return resp;
      });
  app.port(18080).run();
  // if (::bind(fd, reinterpret_cast<struct sockaddr *>(&addr),
  //   sizeof(addr)) != 0) {
  //   printf("bind socket error %d\n", errno);
  //   switch (errno)
  //   {
  //     case EADDRINUSE:
  //       std::cout << "Address already in use" << std::endl;
  //       break;

  //     default:
  //       break;
  //   }
  //   close(fd);
  //   exit(1);
  // }
  // if (listen(fd, 1024) != 0) {
  //   printf("listen socket error %d\n", errno);
  //   close(fd);
  //   exit(1);
  // }
  // while(1) {
  //   struct sockaddr_in cli;
  //   int len = sizeof(cli);
  //   int nfd = accept(fd, reinterpret_cast<struct sockaddr *>(&cli),
  //     reinterpret_cast<socklen_t *>(&len));
  //   if(nfd == -1) {
  //     printf("accept socket error %d\n", errno);
  //     close(fd);
  //     exit(1);
  //   }

  //   char buf[1024];
  //   char content[] = "hello http!";
  //   bzero(buf, 1024);
  //   recv(nfd, buf, 1024, 0);
  //   std::string str(buf);
  //   auto pos = str.find("link=");
  //   auto link = str.substr(pos+5, string::npos);
  //   auto result = Infer(ppocr_v3, link);
  //   printf("----------------recv body----------------\n%s\n", link.c_str());
  //   printf("----------------recv msg----------------\n%s\n", buf);
  //   bzero(buf, 1024);
  //   sprintf(buf, "HTTP/1.1 200 OK\r\nContent-Length: %d\r\n"
  //     "Content-Type: text/html\r\n\r\n%s",
  //     static_cast<int>(strlen(result.c_str())), result.c_str());
  //   printf("----------------ready to send buf----------------\n%s\n", buf);
  //   if (send(nfd, buf, strlen(buf), 0) == -1) {
  //     printf("write socket error %d\n", errno);
  //   }
  //   if (close(nfd) != 0)
  //     printf("close nfd err: %d\n", errno);
  // }

  // if (close(fd) != 0)
  //   printf("close fd err: %d\n", errno);
  // exit(0);
}

int main(int argc, char *argv[]) {
  if (argc < 6) {
    std::cout << "Usage: infer_demo path/to/det_model path/to/cls_model "
                 "path/to/rec_model path/to/rec_label_file path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./ch_PP-OCRv3_det_infer "
                 "./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer "
                 "./ppocr_keys_v1.txt ./12.jpg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, e.g. 0: run with paddle "
                 "inference on cpu;"
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  int flag = std::atoi(argv[5]);

  if (flag == 0) {
    option.UseCpu();
    option.UsePaddleBackend();  // Paddle Inference
  } else if (flag == 1) {
    option.UseCpu();
    option.UseOpenVINOBackend();  // OpenVINO
  } else if (flag == 2) {
    option.UseCpu();
    option.UseOrtBackend();  // ONNX Runtime
  } else if (flag == 3) {
    option.UseCpu();
    option.UseLiteBackend();  // Paddle Lite
  } else if (flag == 4) {
    option.UseGpu();
    option.UsePaddleBackend();  // Paddle Inference
  } else if (flag == 5) {
    option.UseGpu();
    option.UsePaddleInferBackend();
    option.paddle_infer_option.collect_trt_shape = true;
    option.paddle_infer_option.enable_trt = true;  // Paddle-TensorRT
  } else if (flag == 6) {
    option.UseGpu();
    option.UseOrtBackend();  // ONNX Runtime
  } else if (flag == 7) {
    option.UseGpu();
    option.UseTrtBackend();  // TensorRT
  }

  std::string det_model_dir = argv[1];
  std::string cls_model_dir = argv[2];
  std::string rec_model_dir = argv[3];
  std::string rec_label_file = argv[4];
  // std::string test_image = argv[5];
  // InitAndInfer(det_model_dir, cls_model_dir, rec_model_dir, rec_label_file,
  //              test_image, option);

  build(det_model_dir, cls_model_dir, rec_model_dir, rec_label_file, option);
  return 0;
}