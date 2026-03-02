#ifndef PHERONOME_HPP
#define PHERONOME_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <utility>
#include <vector>

#include "basic_types.hpp"

class pheronome {
public:
    using size_t = unsigned long;
    using pheronome_t = std::array<double, 2>;

    pheronome(size_t dim, const position_t& pos_food, const position_t& pos_nest, double alpha = 0.7,
              double beta = 0.9999)
        : m_dim(dim),
          m_stride(dim + 2),
          m_alpha(alpha),
          m_beta(beta),
          m_v1(m_stride * m_stride, 0.0),
          m_v2(m_stride * m_stride, 0.0),
          m_buf_v1(),
          m_buf_v2(),
          m_pos_nest(pos_nest),
          m_pos_food(pos_food) {
        m_v1[index(pos_food)] = 1.0;
        m_v2[index(pos_nest)] = 1.0;
        cl_update_map(m_v1, m_v2);
        m_buf_v1 = m_v1;
        m_buf_v2 = m_v2;
    }

    pheronome(const pheronome&) = delete;
    pheronome(pheronome&&) = delete;
    ~pheronome() = default;

    size_t stride() const { return m_stride; }

    size_t flat_index(int i, int j) const {
        return static_cast<size_t>(i + 1) * m_stride + static_cast<size_t>(j + 1);
    }

    double value(size_t flat, int channel) const { return channel == 0 ? m_v1[flat] : m_v2[flat]; }

    double value(int i, int j, int channel) const { return value(flat_index(i, j), channel); }

    bool blocked(size_t flat, int channel) const { return value(flat, channel) == -1.0; }

    pheronome_t operator()(size_t i, size_t j) const {
        const size_t flat = flat_index(static_cast<int>(i), static_cast<int>(j));
        return {m_v1[flat], m_v2[flat]};
    }

    void do_evaporation() {
        for (std::size_t i = 1; i <= m_dim; ++i) {
            for (std::size_t j = 1; j <= m_dim; ++j) {
                const size_t flat = i * m_stride + j;
                m_buf_v1[flat] *= m_beta;
                m_buf_v2[flat] *= m_beta;
            }
        }
    }

    void mark_pheronome_flat(size_t center_flat) {
        const size_t left = center_flat - m_stride;
        const size_t right = center_flat + m_stride;
        const size_t up = center_flat - 1;
        const size_t down = center_flat + 1;

        const double v1_left = std::max(m_v1[left], 0.0);
        const double v2_left = std::max(m_v2[left], 0.0);
        const double v1_right = std::max(m_v1[right], 0.0);
        const double v2_right = std::max(m_v2[right], 0.0);
        const double v1_up = std::max(m_v1[up], 0.0);
        const double v2_up = std::max(m_v2[up], 0.0);
        const double v1_down = std::max(m_v1[down], 0.0);
        const double v2_down = std::max(m_v2[down], 0.0);

        m_buf_v1[center_flat] =
            m_alpha * std::max({v1_left, v1_right, v1_up, v1_down}) +
            (1.0 - m_alpha) * 0.25 * (v1_left + v1_right + v1_up + v1_down);

        m_buf_v2[center_flat] =
            m_alpha * std::max({v2_left, v2_right, v2_up, v2_down}) +
            (1.0 - m_alpha) * 0.25 * (v2_left + v2_right + v2_up + v2_down);
    }

    void mark_pheronome(int i, int j) {
        assert(i >= 0);
        assert(j >= 0);
        assert(static_cast<size_t>(i) < m_dim);
        assert(static_cast<size_t>(j) < m_dim);
        mark_pheronome_flat(flat_index(i, j));
    }

    void mark_pheronome(const position_t& pos) { mark_pheronome(pos.x, pos.y); }

    void update() {
        m_v1.swap(m_buf_v1);
        m_v2.swap(m_buf_v2);
        cl_update_map(m_v1, m_v2);
        m_v1[index(m_pos_food)] = 1.0;
        m_v2[index(m_pos_nest)] = 1.0;
    }

private:
    size_t index(const position_t& pos) const { return (pos.x + 1) * m_stride + pos.y + 1; }

    void cl_update_map(std::vector<double>& v1, std::vector<double>& v2) {
        for (unsigned long j = 0; j < m_stride; ++j) {
            const size_t top = j;
            const size_t bottom = j + m_stride * (m_dim + 1);
            const size_t left = j * m_stride;
            const size_t right = j * m_stride + m_dim + 1;
            v1[top] = -1.0;
            v2[top] = -1.0;
            v1[bottom] = -1.0;
            v2[bottom] = -1.0;
            v1[left] = -1.0;
            v2[left] = -1.0;
            v1[right] = -1.0;
            v2[right] = -1.0;
        }
    }

    unsigned long m_dim;
    unsigned long m_stride;
    double m_alpha;
    double m_beta;
    std::vector<double> m_v1;
    std::vector<double> m_v2;
    std::vector<double> m_buf_v1;
    std::vector<double> m_buf_v2;
    position_t m_pos_nest;
    position_t m_pos_food;
};

#endif
