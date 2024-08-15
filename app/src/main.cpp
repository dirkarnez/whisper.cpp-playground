#define SDL_MAIN_HANDLED 

// #include <SDL2/SDL.h>
// #include <SDL2/SDL_image.h>
// #include <vector>

// int main(int argc, char* argv[]) {
//     // Initialize SDL
//     SDL_Init(SDL_INIT_VIDEO);
//     IMG_Init(IMG_INIT_PNG);

//     // Create a transparent window
//     SDL_Window* window = SDL_CreateShapedWindow("Clippy Animation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 300, 400, SDL_WINDOW_SHOWN);
    
//     SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0); // SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
//     // SDL_SetWindowOpacity(win, .3);

//     // Load the Clippy image sequence
//     std::vector<SDL_Texture*> clippy_frames;

//     for (int i = 1; i <= 2; i++) {
//         char filename[32];
//         snprintf(filename, sizeof(filename), "clippy_%02d.png", i);
//         SDL_Surface* surface = IMG_Load(filename);
//         SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
//         SDL_FreeSurface(surface);
//         clippy_frames.push_back(texture);
//     }

//         SDL_Surface* surface = IMG_Load("clippy_01.png");
//         SDL_SetWindowShape(window, surface, NULL);
//         // SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
//         // SDL_FreeSurface(surface);
//         // clippy_frames.push_back(texture);

//     // Animation loop
//     bool running = true;
//     int current_frame = 0;
//     while (running) {
//         SDL_Event event;
//         while (SDL_PollEvent(&event)) {
//             if (event.type == SDL_QUIT) {
//                 running = false;
//             }
//         }

//         // Clear the window and draw the current Clippy frame
//         SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
//         SDL_RenderClear(renderer);
//         SDL_Rect dst_rect = {50, 50, 200, 300};
//         SDL_RenderCopy(renderer, clippy_frames[current_frame], NULL, &dst_rect);
//         SDL_RenderPresent(renderer);

//         // Advance to the next frame
//         current_frame = (current_frame + 1) % clippy_frames.size();
//         SDL_Delay(100);
//     }

//     // Clean up
//     for (SDL_Texture* texture : clippy_frames) {
//         SDL_DestroyTexture(texture);
//     }
//     SDL_DestroyRenderer(renderer);
//     SDL_DestroyWindow(window);
//     IMG_Quit();
//     SDL_Quit();

//     return 0;
// }


// #define SDL_MAIN_HANDLED 
// #include <stdio.h>
// #include <SDL2/SDL.h>
// #include <SDL2/SDL_syswm.h>
// #include <windows.h>

// static HWND window_handle(SDL_Window *window) {
// 	SDL_SysWMinfo wmInfo;
// 	SDL_VERSION(&wmInfo.version);
// 	SDL_GetWindowWMInfo(window, &wmInfo);
// 	HWND hwnd = wmInfo.info.win.window;
// 	return hwnd;
// }

// // Any pixel of this color will be rendered as transparent
// static const COLORREF transparent_colorref = RGB(255, 0, 255);

// /**
//  * Returns 0 on failure, 1 on success. Sets up the given window for rendering transparent pixels.
//  */
// int enable_transparency(SDL_Window *window) {
// 	HWND handle = window_handle(window);
// 	if(!SetWindowLong(handle, GWL_EXSTYLE, GetWindowLong(handle, GWL_EXSTYLE) | WS_EX_LAYERED)) {
// 		fprintf(stderr, "SetWindowLong Error\n");
// 		return 0;
// 	}
// 	if(!SetLayeredWindowAttributes(handle, transparent_colorref, 0, 1)) {
// 		fprintf(stderr, "SetLayeredWindowAttributes Error\n");
// 		return 0;
// 	}
// 	return 1;
// }


// /**
//  * Fills the entire renderer with transparent background.
//  */
// void draw_transparent_background(SDL_Renderer *renderer) {
// 	SDL_SetRenderDrawColor(renderer, GetRValue(transparent_colorref), GetGValue(transparent_colorref), GetBValue(transparent_colorref), SDL_ALPHA_OPAQUE);
// 	SDL_RenderFillRect(renderer, NULL);
// }

// int main(int argc, char* argv[]) {
//     // Initialize SDL
//     SDL_Init(SDL_INIT_VIDEO);
//     SDL_Window* window = SDL_CreateShapedWindow("Clippy Animation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 300, 400, SDL_ALPHA_TRANSPARENT);
//     SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

//     draw_transparent_background(renderer);
//     enable_transparency(window);


//     bool running = true;
//     while (running) {
//         SDL_Event event;
//         while (SDL_PollEvent(&event)) {
//             if (event.type == SDL_QUIT) {
//                 running = false;
//             }
//         }

//         // // Clear the window and draw the current Clippy frame
//         // SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
//         // SDL_RenderClear(renderer);
//         // SDL_Rect dst_rect = {50, 50, 200, 300};
//         // SDL_RenderCopy(renderer, clippy_frames[current_frame], NULL, &dst_rect);
//         // SDL_RenderPresent(renderer);

//         // // Advance to the next frame
//         // current_frame = (current_frame + 1) % clippy_frames.size();
//         SDL_Delay(100);
//     }

//     SDL_DestroyRenderer(renderer);
//     SDL_DestroyWindow(window);
//     // IMG_Quit();
//     SDL_Quit();

//     return 0;
// }

#define SDL_MAIN_HANDLED 
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <vector>

#include <windows.h>

int main(int argc, char *argv[])
{
    IMG_Init(IMG_INIT_PNG);

    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Surface *winSurface;
    SDL_Event event;
    int quit = 0;
    SDL_Rect rectRect = {0,0,500,500};
    SDL_Rect backRect = {0};
    SDL_SysWMinfo info;
    HWND hwnd;

    // Load the Clippy image sequence
    std::vector<SDL_Texture*> clippy_frames;
    int current_frame = 0;

    for (int i = 1; i <= 2; i++) {
        char filename[32];
        snprintf(filename, sizeof(filename), "clippy_%02d.png", i);
        SDL_Surface* surface = IMG_Load(filename);
        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
        SDL_FreeSurface(surface);
        clippy_frames.push_back(texture);
    }

    SDL_Init(SDL_INIT_EVERYTHING);

    window   = SDL_CreateWindow("Test",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,640,480,SDL_WINDOW_SHOWN|SDL_WINDOW_BORDERLESS);
    SDL_VERSION(&info.version);
    if(SDL_GetWindowWMInfo(window,&info))
    {
        hwnd = info.info.win.window;
    }
/*设置窗口colorkey*/
    SetWindowLong( hwnd, GWL_EXSTYLE, GetWindowLong(hwnd, GWL_EXSTYLE)|WS_EX_LAYERED);
    SetLayeredWindowAttributes( hwnd, RGB(255,255,255),0, LWA_COLORKEY);
/*设置窗口为悬浮窗 */
    SetWindowPos ( hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE|SWP_NOSIZE);
/*--------------*/
    winSurface = SDL_GetWindowSurface(window);
    SDL_GetWindowSize(window,&backRect.w,&backRect.h);
    UINT32 keyColor = SDL_MapRGB(winSurface->format,255,255,255); //black
    SDL_SetSurfaceBlendMode(winSurface, SDL_BLENDMODE_NONE);
    

    while(!quit){
        while(SDL_PollEvent(&event))
        {
            if(event.type == SDL_QUIT)
                quit = 1;
        }
      
        // // 

        SDL_FillRect(winSurface, &backRect, keyColor);
        rectRect.x+=1;

        SDL_RenderClear(renderer);
        SDL_RenderPresent(renderer);

        SDL_SetRenderDrawColor(renderer, 255,255,255, 0);
        SDL_FillRect(winSurface, &rectRect, SDL_MapRGB(winSurface->format,0xff,0x00,0x00));
        SDL_RenderCopy(renderer, clippy_frames[current_frame], NULL, &rectRect);
        
// rectRect.w = rectRect.h = 500;
        

        // Advance to the next frame
        current_frame = (current_frame + 1) % clippy_frames.size();
        SDL_UpdateWindowSurface(window);
        SDL_Delay(1000/60);




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