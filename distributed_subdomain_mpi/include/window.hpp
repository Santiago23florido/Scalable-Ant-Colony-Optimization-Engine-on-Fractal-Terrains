#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <SDL2/SDL.h>
#include <utility>

class Window {
public:
    Window(const char* title, int width, int height);
    Window(const Window&) = delete;
    Window(Window&&) = delete;
    ~Window();

    Window& operator=(const Window&) = delete;
    Window& operator=(Window&&) = delete;

    SDL_Window* get() { return m_window; }

    void set_pen(Uint8 r, Uint8 g, Uint8 b) { SDL_SetRenderDrawColor(SDL_GetRenderer(m_window), r, g, b, 255); }

    void pset(int x, int y) { SDL_RenderDrawPoint(SDL_GetRenderer(m_window), x, y); }

    void blit() { SDL_RenderPresent(SDL_GetRenderer(m_window)); }

    std::pair<int, int> size() {
        int w = 0;
        int h = 0;
        SDL_GetWindowSize(m_window, &w, &h);
        return {w, h};
    }

private:
    SDL_Window* m_window{nullptr};
    SDL_Renderer* m_renderer{nullptr};
};

#endif
