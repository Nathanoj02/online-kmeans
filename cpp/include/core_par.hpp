#ifndef SIGNALS_CORE_PAR_HPP
#define SIGNALS_CORE_PAR_HPP

#include <opencv2/imgcodecs.hpp>

namespace signals {

namespace cuda {

struct DimInfo {
    std::size_t x;
    std::size_t y;
    std::size_t z;
};

struct DeviceInfo {
    DimInfo block;
    DimInfo grid;
};

/**
 * Find best grid for current GPU
 * 
 * @param device_info Struct in which the best block and grid dimensions are memorized 
 * @param height Height of the input matrix
 * @param width Width of the input matrix
 * 
 * @return Device info with best block and grid dimensions
 */
DeviceInfo& find_best_grid(
    DeviceInfo& device_info, std::size_t height, std::size_t width
);

/**
 * K-means clustering algorithm
 * 
 * @param dst Destination image
 * @param img Source image
 * @param img_height Source image height
 * @param img_width Source image width
 * @param k Number of clusters
 * @param stab_error Error bound to reach to end the algorithm
 */
void k_means(
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    uint64_t k, float_t stab_error
);



} // namespace cuda

} // namespace signals

#endif // SIGNALS_CORE_PAR_HPP