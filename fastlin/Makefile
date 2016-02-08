
CC = gcc
FF = gfortran
RM = rm
CFLAGS  = -g -Wall
FFLAGS  = -march=native -Wall -Wextra -O3 -fPIC

default: lib

lib:  dpo2ft.o dpofrt.o choleskyGrad.o
	ar rcs libcholgrad.a choleskyGrad.o dpo2ft.o dpofrt.o

dpo2ft.o: dpo2ft.f
	$(FF) $(FFLAGS) -c dpo2ft.f

dpofrt.o: dpofrt.f
	$(FF) $(FFLAGS) -c dpofrt.f

choleskyGrad.o: choleskyGrad.c choleskyGrad.h
	$(CC) $(CFLAGS) -c choleskyGrad.c

clean: 
	$(RM) *.o *~ *.a