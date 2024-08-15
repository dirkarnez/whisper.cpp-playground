#define SDL_MAIN_HANDLED 

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <vector>

int main(int argc, char* argv[]) {
    // Initialize SDL
    SDL_Init(SDL_INIT_VIDEO);
    IMG_Init(IMG_INIT_PNG);

    // Create a transparent window
    SDL_Window* window = SDL_CreateWindow("Clippy Animation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 300, 400, SDL_ALPHA_TRANSPARENT);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    // Load the Clippy image sequence
    std::vector<SDL_Texture*> clippy_frames;
    for (int i = 1; i <= 12; i++) {
        char filename[32];
        snprintf(filename, sizeof(filename), "clippy_%02d.png", i);
        SDL_Surface* surface = IMG_Load(filename);
        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
        SDL_FreeSurface(surface);
        clippy_frames.push_back(texture);
    }

    // Animation loop
    bool running = true;
    int current_frame = 0;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        // Clear the window and draw the current Clippy frame
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
        SDL_RenderClear(renderer);
        SDL_Rect dst_rect = {50, 50, 200, 300};
        SDL_RenderCopy(renderer, clippy_frames[current_frame], NULL, &dst_rect);
        SDL_RenderPresent(renderer);

        // Advance to the next frame
        current_frame = (current_frame + 1) % clippy_frames.size();
        SDL_Delay(100);
    }

    // Clean up
    for (SDL_Texture* texture : clippy_frames) {
        SDL_DestroyTexture(texture);
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    return 0;
}
