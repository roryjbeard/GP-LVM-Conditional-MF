
#include "choleskyGrad.h"

int main(void){

	double A[] = {1.0024, 0.6825, 0.7596, 0.0, 0.4387, -0.2127, 0.0, 0.0, 0.1794};
	double B[] = {0.5500, 0.9977, 0.7533, 0.0, 0.8551,  0.1701, 0.0, 0.0, 0.6882};

	choleskyGrad( 'L', A, B, 3);

	return 0;
}


