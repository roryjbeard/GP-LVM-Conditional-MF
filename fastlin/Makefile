
CC = gcc
FF = gfortran
RM = rm
CFLAGS  = -g -Wall -fPIC
FFLAGS  = -march=native -Wall -Wextra -O3 -fPIC -mno-avx
# MAC OS
MACSHARE = -dynamiclib
MACLIBS = -framework Accelerate
# UBUNTU
LINSHARE = -shared
LINLIBS = -llapack -lblas

default: choleskyGradTest

# uncomment for linux
libcholgrad.so: dpo2ft.o dpofrt.o choleskyGrad.o
	$(CC) $(LINSHARE) -o libcholgrad.so dpo2ft.o dpofrt.o choleskyGrad.o $(LINLIBS)

# Static library for linux (mac doesn't do static libraries)
#libcholgrad.a: dpo2ft.o dpofrt.o choleskyGrad.o
#	ar rcs tmp.a choleskyGrad.o dpo2ft.o dpofrt.o choleskyGrad.o
#	libtool --mode=link cc -static -o lib.la libabc.la libxyz.la

# Uncomment for mac
#libcholgrad.dylib: dpo2ft.o dpofrt.o choleskyGrad.o
#	$(CC) $(MACSHARE) -o libcholgrad.dylib dpo2ft.o dpofrt.o choleskyGrad.o $(MACLIBS)

dpo2ft.o: dpo2ft.f
	$(FF) $(FFLAGS) -c dpo2ft.f

dpofrt.o: dpofrt.f
	$(FF) $(FFLAGS) -c dpofrt.f

choleskyGrad.o: choleskyGrad.c choleskyGrad.h
	$(CC) $(CFLAGS) -c choleskyGrad.c

# Change libcholgrad.so to libcholgrad.dylib for linux/max 
choleskyGradTest: libcholgrad.so choleskyGradTest.c
	$(CC) $(CFLAGS) choleskyGradTest.c -o choleskyGradTest -L. -lcholgrad 

clean: 
	$(RM) *.o *.so *.dylib choleskyGradTest
