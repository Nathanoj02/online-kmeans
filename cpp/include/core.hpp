#ifndef SIGNALS_CORE_HPP
#define SIGNALS_CORE_HPP

#include <opencv2/imgcodecs.hpp>

namespace signals
{

void print();

void k_means (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    uint64_t k, float_t stab_error);

} // namespace signals

#endif // SIGNALS_CORE_HPP