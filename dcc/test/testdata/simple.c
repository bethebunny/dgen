int add(int a, int b) {
    return a + b;
}

int factorial(int n) {
    int result = 1;
    int i = 1;
    while (i <= n) {
        result = result * i;
        i = i + 1;
    }
    return result;
}

double average(double x, double y) {
    return (x + y) / 2.0;
}
