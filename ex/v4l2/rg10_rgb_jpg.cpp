#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <iostream>

void apply_white_balance(cv::Mat& bayer_image) {
    // Bayer 이미지에서 각 채널의 평균 계산
    int width = bayer_image.cols;
    int height = bayer_image.rows;

    double sum_r = 0, sum_g = 0, sum_b = 0;
    int count_r = 0, count_g = 0, count_b = 0;

    // Bayer RGGB 패턴 기반으로 각 채널의 평균 계산
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint16_t pixel = bayer_image.at<uint16_t>(y, x);
            if (y % 2 == 0 && x % 2 == 0) {  // Red 채널
                sum_r += pixel;
                count_r++;
            } else if (y % 2 == 0 && x % 2 == 1) {  // Green 채널 (Red Row)
                sum_g += pixel;
                count_g++;
            } else if (y % 2 == 1 && x % 2 == 0) {  // Green 채널 (Blue Row)
                sum_g += pixel;
                count_g++;
            } else if (y % 2 == 1 && x % 2 == 1) {  // Blue 채널
                sum_b += pixel;
                count_b++;
            }
        }
    }

    // 평균 값 계산
    double avg_r = sum_r / count_r;
    double avg_g = sum_g / count_g;
    double avg_b = sum_b / count_b;

    // 게인 계산 (Green을 기준으로 정규화)
    double gain_r = avg_g / avg_r;
    double gain_b = avg_g / avg_b;

    std::cout << "White balance gains: R=" << gain_r << ", G=1.0, B=" << gain_b << std::endl;

    // 각 채널에 게인 적용
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint16_t& pixel = bayer_image.at<uint16_t>(y, x);
            if (y % 2 == 0 && x % 2 == 0) {  // Red 채널
                pixel = cv::saturate_cast<uint16_t>(pixel * gain_r);
            } else if (y % 2 == 1 && x % 2 == 1) {  // Blue 채널
                pixel = cv::saturate_cast<uint16_t>(pixel * gain_b);
            }
            // Green 채널은 그대로 유지
        }
    }
}

// SSIM 계산 함수
cv::Scalar computeSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    const double C1 = 6.5025, C2 = 58.5225;

    cv::Mat img1_32f, img2_32f;
    img1.convertTo(img1_32f, CV_32F);
    img2.convertTo(img2_32f, CV_32F);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1_32f, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_32f, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_mu2 = mu1.mul(mu2);
    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);

    cv::Mat sigma1, sigma2;
    cv::GaussianBlur(img1_32f.mul(img1_32f), sigma1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_32f.mul(img2_32f), sigma2, cv::Size(11, 11), 1.5);

    sigma1 -= mu1_sq;
    sigma2 -= mu2_sq;

    cv::Mat sigma12;
    cv::GaussianBlur(img1_32f.mul(img2_32f), sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1 = 2 * mu1_mu2 + C1;
    cv::Mat t2 = 2 * sigma12 + C2;
    cv::Mat t3 = t1.mul(t2);

    t1 = mu1_sq + mu2_sq + C1;
    t2 = sigma1 + sigma2 + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map;
    divide(t3, t1, ssim_map);
    cv::Scalar mssim = mean(ssim_map);
    return mssim;
}

// RAW 파일을 처리하여 RGB 이미지로 변환하는 함수
cv::Mat processRawImage(const std::string& filename, int width, int height) {
    // RAW 파일 열기
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // RG10 데이터를 읽어서 16비트 배열로 저장
    std::vector<uint16_t> buffer(width * height);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(uint16_t));
    file.close();

    // Bayer 데이터를 OpenCV Mat로 변환
    cv::Mat bayer_image_16bit(height, width, CV_16UC1, buffer.data());

    // 화이트 밸런스 적용(CV_16U1)
    apply_white_balance(bayer_image_16bit);

    // 이미지 정보 출력
    std::cout << "Image size: " << bayer_image_16bit.cols << "x" << bayer_image_16bit.rows << std::endl;
    std::cout << "Image type 0(CV_8U), 1(CV_8S), 2(CV_16U): " << bayer_image_16bit.depth() << std::endl;
    std::cout << "Number of channels(C1,C3): " << bayer_image_16bit.channels() << std::endl;
    double min_val, max_val;
    cv::minMaxLoc(bayer_image_16bit, &min_val, &max_val);
    std::cout << "Min pixel value: " << min_val << ", Max pixel value: " << max_val << std::endl;

    // 정규화 수행 (CV_16UC1 → CV_8UC1)
    cv::Mat normalized_image;
    bayer_image_16bit.convertTo(normalized_image, CV_8UC1, 255.0 / max_val);
    std::cout << "Nomalized Image size: " << normalized_image.cols << "x" << normalized_image.rows << std::endl;
    std::cout << "INomalized Image depth: " << normalized_image.depth() << std::endl;
    std::cout << "NNomalized Number of channels: " << normalized_image.channels() << std::endl;
    cv::minMaxLoc(normalized_image, &min_val, &max_val);
    std::cout << "Nomalized Min pixel value: " << min_val << ", Nomalized Max pixel value: " << max_val << std::endl;


    // Bayer(CV_8UC1) -> RGB 변환 (CV_8UC3) 디바이커링
    cv::Mat rgb_image, rgb_image_vng, rgb_image_ea;
    cv::cvtColor(normalized_image, rgb_image, cv::COLOR_BayerRG2RGB);  // 디마이커링
    cv::cvtColor(normalized_image, rgb_image_vng, cv::COLOR_BayerRG2RGB_VNG);  // 디마이커링
    cv::cvtColor(normalized_image, rgb_image_ea, cv::COLOR_BayerRG2RGB_EA);  // 디마이커링


    // 품질 평가
    double psnr = cv::PSNR(rgb_image_vng,rgb_image);
    std::cout << "PSNR(vng,rgb): " << psnr << std::endl;

    psnr = cv::PSNR(rgb_image_vng,rgb_image_ea);
    std::cout << "PSNR(vng,ea): " << psnr << std::endl;

    psnr = cv::PSNR(rgb_image_ea,rgb_image);
    std::cout << "PSNR(ea,rgb): " << psnr << std::endl;

    // SSIM 계산
    cv::Scalar ssim = computeSSIM(rgb_image, rgb_image_vng);
    std::cout << "SSIM(rgb,vng): " << ssim[0] << std::endl;

    ssim = computeSSIM(rgb_image, rgb_image_ea);
    std::cout << "SSIM(rgb,ea): " << ssim[0] << std::endl;

    ssim = computeSSIM(rgb_image_ea, rgb_image_vng);
    std::cout << "SSIM(ea,vng): " << ssim[0] << std::endl;


    // 이미지 표시 및 저장
    cv::imshow(" normalized Image", normalized_image);
    cv::imwrite("Normalized_Image.png", normalized_image);

    cv::waitKey(0);

    cv::imshow(" rgb Image", rgb_image);
    cv::imwrite("rgb_Image.png", rgb_image);

    cv::waitKey(0);


    cv::imshow("vng Image", rgb_image_vng);
    cv::imwrite("vng_Image.png", rgb_image_vng);

    cv::waitKey(0);

    cv::imshow("ea Image", rgb_image_ea);
    cv::imwrite("EA_Image.png", rgb_image_ea);


    cv::waitKey(0);


    return rgb_image_ea;
}

// 색 보정 행렬(CCM) 적용 함수
cv::Mat applyCCM(const cv::Mat& rgb_image) {
    // IMX219용 기본 CCM (필요 시 데이터시트에 맞게 수정)
    cv::Mat ccm = (cv::Mat_<float>(3, 3) <<
        1.2, -0.1, -0.1,
        -0.1, 1.1,  0.0,
        -0.1,  0.0,  1.3
    );

    // CCM 적용
    cv::Mat corrected_image;
    cv::transform(rgb_image, corrected_image, ccm);

    return corrected_image;
}

// 감마 보정 함수
cv::Mat applyGammaCorrection(const cv::Mat& image, double gamma) {
    cv::Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++) {
        lut.at<uchar>(i) = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    cv::Mat corrected_image;
    cv::LUT(image, lut, corrected_image);
    return corrected_image;
}

int main() {
    try {
        int width = 3280;   // 이미지 폭
        int height = 2464;  // 이미지 높이

        // RAW 파일을 처리하여 RGB 이미지 생성
        cv::Mat rgb_image = processRawImage("frame.raw", width, height);

        // 색 보정 행렬 적용
        //cv::Mat ccm_image = applyCCM(balanced_image);

        // 감마 보정 적용
        //cv::Mat gamma_corrected_image = applyGammaCorrection(ccm_image, 1 / 2.2); // 감마 값은 2.2의 역수

        // 결과 저장 및 표시
        //cv::imwrite("final_image.png", gamma_corrected_image);
        //cv::imshow("Final Image", gamma_corrected_image);
        //cv::waitKey(0);

        //std::cout << "Image processing completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

