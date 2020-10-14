#include <cstdlib>
#include <cstdio>

float uinttofp32(unsigned int a) {
    float* b = (float*)&a;
    int b_s = ((rand()%2) * 2 - 1);
    return (*b) * b_s;
}

float fp32_denormal() {
    unsigned int a = rand() & 0x7fffff;
    return uinttofp32(a);
}

int main() {
    int len = 10;
    for (int i = 0; i < len; ++i) {
        printf("%e\n", fp32_denormal());
    }
    return 0;
}
