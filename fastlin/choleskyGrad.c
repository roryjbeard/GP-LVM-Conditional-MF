
#include <stdint.h>
#include "choleskyGrad.h"

extern void dpofrt_(
        char *uplo, const int32_t *n, double *a, const int32_t *lda,
        double *c, const int32_t *ldc, const int32_t *info);

int choleskyGrad(char UPLO, double* L, double* F, int N) {
    
    int32_t info;

    dpofrt_(&UPLO, &N, L, &N, F, &N, &info);
    
    return info;
}
