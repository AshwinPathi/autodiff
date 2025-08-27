#pragma once

#include <cassert>
#define assertm(exp, msg) assert((void(msg), exp))

template <typename T>
    requires(sizeof(T) == 4)
inline constexpr T swap_bytes_32bit(T in) {
    uint8_t p1 = in & 0x000000FF;
    uint8_t p2 = (in & 0x0000FF00) >> 8;
    uint8_t p3 = (in & 0x00FF0000) >> 16;
    uint8_t p4 = (in & 0xFF000000) >> 24;

    return (static_cast<T>(p1) << 24) | (static_cast<T>(p2) << 16) | (static_cast<T>(p3) << 8) |
           (static_cast<T>(p4));
}
