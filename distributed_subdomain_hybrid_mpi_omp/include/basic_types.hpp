#ifndef BASIC_TYPES_HPP
#define BASIC_TYPES_HPP

#include <SDL2/SDL.h>
#include <utility>

using position_t = SDL_Point;

inline bool operator==(const position_t& lhs, const position_t& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

using dimension_t = std::pair<std::size_t, std::size_t>;

#endif
