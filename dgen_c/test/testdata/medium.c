typedef unsigned int uint32_t;
typedef unsigned long size_t;
typedef int int32_t;

struct Point {
    double x;
    double y;
};

enum Color {
    RED = 0,
    GREEN = 1,
    BLUE = 2,
};

double distance(struct Point *a, struct Point *b) {
    double dx = a->x - b->x;
    double dy = a->y - b->y;
    return dx * dx + dy * dy;
}

int fibonacci(int n) {
    if (n <= 1)
        return n;
    int a = 0;
    int b = 1;
    int i;
    for (i = 2; i <= n; i++) {
        int tmp = a + b;
        a = b;
        b = tmp;
    }
    return b;
}

void bubble_sort(int *arr, int n) {
    int i, j;
    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

uint32_t hash(const char *str) {
    uint32_t h = 5381;
    int c;
    while ((c = *str++) != 0) {
        h = h * 33 + c;
    }
    return h;
}

int binary_search(int *arr, int n, int target) {
    int low = 0;
    int high = n - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return -1;
}
