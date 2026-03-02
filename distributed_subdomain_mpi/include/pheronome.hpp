#ifndef PHERONOME_HPP
#define PHERONOME_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

#include "basic_types.hpp"
#include "subdomain.hpp"

class pheronome {
public:
    using size_t = unsigned long;
    using pheronome_t = std::array<double, 2>;

    pheronome(size_t dim, const position_t& pos_food, const position_t& pos_nest, double alpha = 0.7, double beta = 0.9999)
        : pheronome(static_cast<int>(dim), static_cast<int>(dim), static_cast<int>(dim), static_cast<int>(dim), 0, 0, 1, false, false, false, false, pos_food, pos_nest, alpha, beta) {}

    pheronome(const mpi_subdomain::DomainDecomposition& decomp, const position_t& pos_food, const position_t& pos_nest, double alpha = 0.7, double beta = 0.9999)
        : pheronome(decomp.global_nx, decomp.global_ny, decomp.local_nx, decomp.local_ny, decomp.offset_x, decomp.offset_y, decomp.ghost, decomp.left_rank != MPI_PROC_NULL, decomp.right_rank != MPI_PROC_NULL, decomp.up_rank != MPI_PROC_NULL, decomp.down_rank != MPI_PROC_NULL, pos_food, pos_nest, alpha, beta) {}

    pheronome(const pheronome&) = delete;
    pheronome(pheronome&&) = delete;
    ~pheronome() = default;

    int global_nx() const { return m_global_nx; }
    int global_ny() const { return m_global_ny; }
    int local_nx() const { return m_local_nx; }
    int local_ny() const { return m_local_ny; }
    int offset_x() const { return m_offset_x; }
    int offset_y() const { return m_offset_y; }
    int ghost() const { return m_ghost; }
    int stride_x() const { return m_stride_x; }
    int stride_y() const { return m_stride_y; }
    size_t halo_size() const { return static_cast<size_t>(m_stride_x) * static_cast<size_t>(m_stride_y); }
    size_t stride() const { return static_cast<size_t>(m_stride_x); }
    bool owns_global(int gx, int gy) const { return gx >= m_offset_x && gx < m_offset_x + m_local_nx && gy >= m_offset_y && gy < m_offset_y + m_local_ny; }
    int local_x_from_global(int gx) const { return (gx - m_offset_x) + m_ghost; }
    int local_y_from_global(int gy) const { return (gy - m_offset_y) + m_ghost; }
    size_t idx_local(int local_x, int local_y) const { return static_cast<size_t>(local_y) * static_cast<size_t>(m_stride_x) + static_cast<size_t>(local_x); }
    size_t flat_index(int i, int j) const { return idx_local(i + m_ghost, j + m_ghost); }
    size_t idx_global(int gx, int gy) const { return idx_local(local_x_from_global(gx), local_y_from_global(gy)); }
    std::vector<double>& current_channel(int channel) { return channel == 0 ? m_v1 : m_v2; }
    const std::vector<double>& current_channel(int channel) const { return channel == 0 ? m_v1 : m_v2; }
    std::vector<double>& buffer_channel(int channel) { return channel == 0 ? m_buf_v1 : m_buf_v2; }
    const std::vector<double>& buffer_channel(int channel) const { return channel == 0 ? m_buf_v1 : m_buf_v2; }
    void copy_current_to_buffer() { m_buf_v1 = m_v1; m_buf_v2 = m_v2; }
    void swap_current_with_buffer() { m_v1.swap(m_buf_v1); m_v2.swap(m_buf_v2); }
    void set_current_global_value(int gx, int gy, int channel, double value) { if (owns_global(gx, gy)) current_channel(channel)[idx_global(gx, gy)] = value; }
    void set_buffer_global_value(int gx, int gy, int channel, double value) { if (owns_global(gx, gy)) buffer_channel(channel)[idx_global(gx, gy)] = value; }
    void set_current_physical_ghosts(double boundary_value = -1.0) { apply_physical_ghosts(m_v1, m_v2, boundary_value); }
    void set_buffer_physical_ghosts(double boundary_value = -1.0) { apply_physical_ghosts(m_buf_v1, m_buf_v2, boundary_value); }
    double read_global(int gx, int gy, int channel, double outside_value = -1.0) const { return read_global_channel(current_channel(channel), gx, gy, outside_value); }
    double read_buffer_global(int gx, int gy, int channel, double outside_value = -1.0) const { return read_global_channel(buffer_channel(channel), gx, gy, outside_value); }

    double value(size_t flat, int channel) const { return current_channel(channel)[flat]; }
    double value(int i, int j, int channel) const { return value(flat_index(i, j), channel); }
    bool blocked(size_t flat, int channel) const { return value(flat, channel) == -1.0; }
    pheronome_t operator()(size_t i, size_t j) const { const size_t flat = flat_index(static_cast<int>(i), static_cast<int>(j)); return {m_v1[flat], m_v2[flat]}; }
    void do_evaporation() { for (int j = 0; j < m_local_ny; ++j) for (int i = 0; i < m_local_nx; ++i) { const size_t flat = flat_index(i, j); m_buf_v1[flat] *= m_beta; m_buf_v2[flat] *= m_beta; } }

    void mark_pheronome_flat(size_t center_flat) {
        const size_t left = center_flat - 1;
        const size_t right = center_flat + 1;
        const size_t up = center_flat - static_cast<size_t>(m_stride_x);
        const size_t down = center_flat + static_cast<size_t>(m_stride_x);
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

    void mark_pheronome(int i, int j) { assert(i >= 0); assert(j >= 0); assert(i < m_local_nx); assert(j < m_local_ny); mark_pheronome_flat(flat_index(i, j)); }
    void mark_pheronome(const position_t& pos) { mark_pheronome(pos.x, pos.y); }

    void load_from_dense(const std::vector<double>& dense_v1, const std::vector<double>& dense_v2) {
        assert(dense_v1.size() == static_cast<size_t>(m_local_nx * m_local_ny));
        assert(dense_v2.size() == static_cast<size_t>(m_local_nx * m_local_ny));
        for (int j = 0; j < m_local_ny; ++j) {
            for (int i = 0; i < m_local_nx; ++i) {
                const size_t dense_idx = static_cast<size_t>(j * m_local_nx + i);
                const size_t flat = flat_index(i, j);
                m_v1[flat] = dense_v1[dense_idx];
                m_v2[flat] = dense_v2[dense_idx];
            }
        }
        set_current_physical_ghosts(-1.0);
        enforce_sources(m_v1, m_v2);
        m_buf_v1 = m_v1;
        m_buf_v2 = m_v2;
    }

    void update() {
        swap_current_with_buffer();
        set_current_physical_ghosts(-1.0);
        enforce_sources(m_v1, m_v2);
    }

private:
    pheronome(int global_nx, int global_ny, int local_nx, int local_ny, int offset_x, int offset_y, int ghost, bool has_left_neighbor, bool has_right_neighbor, bool has_up_neighbor, bool has_down_neighbor, const position_t& pos_food, const position_t& pos_nest, double alpha, double beta)
        : m_global_nx(global_nx),
          m_global_ny(global_ny),
          m_local_nx(local_nx),
          m_local_ny(local_ny),
          m_offset_x(offset_x),
          m_offset_y(offset_y),
          m_ghost(ghost),
          m_stride_x(local_nx + 2 * ghost),
          m_stride_y(local_ny + 2 * ghost),
          m_alpha(alpha),
          m_beta(beta),
          m_v1(static_cast<size_t>(m_stride_x) * static_cast<size_t>(m_stride_y), 0.0),
          m_v2(static_cast<size_t>(m_stride_x) * static_cast<size_t>(m_stride_y), 0.0),
          m_buf_v1(static_cast<size_t>(m_stride_x) * static_cast<size_t>(m_stride_y), 0.0),
          m_buf_v2(static_cast<size_t>(m_stride_x) * static_cast<size_t>(m_stride_y), 0.0),
          m_pos_nest(pos_nest),
          m_pos_food(pos_food),
          m_has_left_neighbor(has_left_neighbor),
          m_has_right_neighbor(has_right_neighbor),
          m_has_up_neighbor(has_up_neighbor),
          m_has_down_neighbor(has_down_neighbor) {
        set_current_physical_ghosts(-1.0);
        enforce_sources(m_v1, m_v2);
        copy_current_to_buffer();
    }

    bool in_global_bounds(int gx, int gy) const { return gx >= 0 && gx < m_global_nx && gy >= 0 && gy < m_global_ny; }
    double read_global_channel(const std::vector<double>& channel, int gx, int gy, double outside_value) const {
        if (!in_global_bounds(gx, gy)) return outside_value;
        const int lx = local_x_from_global(gx);
        const int ly = local_y_from_global(gy);
        if (lx < 0 || lx >= m_stride_x || ly < 0 || ly >= m_stride_y) return outside_value;
        return channel[idx_local(lx, ly)];
    }
    void apply_physical_ghosts(std::vector<double>& v1, std::vector<double>& v2, double boundary_value) {
        if (!m_has_left_neighbor) for (int ly = 0; ly < m_stride_y; ++ly) { v1[idx_local(0, ly)] = boundary_value; v2[idx_local(0, ly)] = boundary_value; }
        if (!m_has_right_neighbor) for (int ly = 0; ly < m_stride_y; ++ly) { v1[idx_local(m_local_nx + m_ghost, ly)] = boundary_value; v2[idx_local(m_local_nx + m_ghost, ly)] = boundary_value; }
        if (!m_has_up_neighbor) for (int lx = 0; lx < m_stride_x; ++lx) { v1[idx_local(lx, 0)] = boundary_value; v2[idx_local(lx, 0)] = boundary_value; }
        if (!m_has_down_neighbor) for (int lx = 0; lx < m_stride_x; ++lx) { v1[idx_local(lx, m_local_ny + m_ghost)] = boundary_value; v2[idx_local(lx, m_local_ny + m_ghost)] = boundary_value; }
    }
    void enforce_sources(std::vector<double>& v1, std::vector<double>& v2) {
        if (owns_global(m_pos_food.x, m_pos_food.y)) v1[idx_global(m_pos_food.x, m_pos_food.y)] = 1.0;
        if (owns_global(m_pos_nest.x, m_pos_nest.y)) v2[idx_global(m_pos_nest.x, m_pos_nest.y)] = 1.0;
    }

    int m_global_nx;
    int m_global_ny;
    int m_local_nx;
    int m_local_ny;
    int m_offset_x;
    int m_offset_y;
    int m_ghost;
    int m_stride_x;
    int m_stride_y;
    double m_alpha;
    double m_beta;
    std::vector<double> m_v1;
    std::vector<double> m_v2;
    std::vector<double> m_buf_v1;
    std::vector<double> m_buf_v2;
    position_t m_pos_nest;
    position_t m_pos_food;
    bool m_has_left_neighbor;
    bool m_has_right_neighbor;
    bool m_has_up_neighbor;
    bool m_has_down_neighbor;
};

#endif
