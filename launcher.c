#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mach-o/dyld.h>
#include <limits.h>

int main(int argc, char *argv[]) {
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    
    // Get the path of the current executable
    if (_NSGetExecutablePath(path, &size) == 0) {
        char *last_slash = strrchr(path, '/');
        if (last_slash) {
            *last_slash = '\0'; // path now points to Contents/MacOS
            char setup_script[PATH_MAX];
            snprintf(setup_script, sizeof(setup_script), "%s/../Resources/setup.sh", path);
            
            // Execute the setup script, replacing the current process
            char *args[] = {"/bin/zsh", setup_script, NULL};
            execv("/bin/zsh", args);
            
            // If execv fails
            perror("execv failed");
            return 1;
        }
    }
    return 1;
}
