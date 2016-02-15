
CC = gcc
FF = gfortran
RM = rm
CFLAGS  = -g -Wall -fPIC
# MAC OS
#FFLAGS  = -march=native -mno-avx -Wall -Wextra -O3 -fPIC
#LIBFLAGS = -dynamiclib
#LIBS = -framework Accelerate
# UBUNTU
FFLAGS  = -march=native -Wall -Wextra -O3 -fPIC
LIBFLAGS = -shared
LIBS = -llapack -lblas

default: choleskyGradTest

libcholgrad: dpo2ft.o dpofrt.o choleskyGrad.o
	$(CC) $(LIBFLAGS) -o libcholgrad.so dpo2ft.o dpofrt.o choleskyGrad.o $(LIBS)
dpo2ft.o: dpo2ft.f
	$(FF) $(FFLAGS) -c dpo2ft.f

dpofrt.o: dpofrt.f
	$(FF) $(FFLAGS) -c dpofrt.f

choleskyGrad.o: choleskyGrad.c choleskyGrad.h
	$(CC) $(CFLAGS) -c choleskyGrad.c

choleskyGradTest: libcholgrad choleskyGradTest.c
	$(CC) $(CFLAGS) choleskyGradTest.c -o choleskyGradTest -L. -lcholgrad 

clean: 
	$(RM) *.o *.so *.dylib choleskyGradTest
