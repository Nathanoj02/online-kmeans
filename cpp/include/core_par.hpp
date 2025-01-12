#ifndef SIGNALS_CORE_PAR_HPP
#define SIGNALS_CORE_PAR_HPP

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

DeviceInfo& find_best_grid(
    DeviceInfo& device_info, std::size_t height, std::size_t width
);


// void k_means (
//     uint8_t* dst, uint8_t* img,
//     size_t img_height, size_t img_width,
//     uint64_t k, float_t stab_error
// );

void print();






} // namespace cuda

} // namespace signals

#endif // SIGNALS_CORE_PAR_HPP