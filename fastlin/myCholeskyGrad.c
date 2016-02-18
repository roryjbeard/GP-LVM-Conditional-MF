

#section support_code

// Support code function
bool vector_same_shape(PyArrayObject* arr1, PyArrayObject* arr2) {
    return (PyArray_DIMS(arr1)[0] == PyArray_DIMS(arr2)[0] && PyArray_DIMS(arr1)[1] == PyArray_DIMS(arr2)[1]);
}

void printMatrix(std::string name, npy_float64* ptr, int str0, int str1, int N) {
    std::cout << name << "[=\n";
    for(int r = 0; r < N; ++r) {
        std::cout << "[" << ptr[r * str0]; 
        for(int c = 1; c < N; ++c) {
            std::cout << ", " << ptr[r * str0 + c * str1];
        }
        std::cout << "]\n"; 
    }
    std::cout << "]\n";
}

void tril(npy_float64* ptr, int str0, int str1, int N) {
    for(int c = 0; c < N; ++c) {
        for(int r = 0; r < c; ++r) {
            ptr[r * str0 + c * str1] = 0.0;
        }
    }
}

void triu(npy_float64* ptr, int str0, int str1, int N)
{
    for(int c = 0; c < N; ++c){
        for(int r = c+1; r < N; ++r) {
            ptr[r * str0 + c * str1] = 0.0;
        }
    }
}

#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(apply_cholesky_grad)(PyArrayObject* input0,
                                        PyArrayObject* input1,
                                        PyArrayObject* input2,
                                        PyArrayObject** output0)
{
    int N, rCode = 0;
    PyArrayObject *L=NULL, *F=NULL;

    // Validate that the inputs have the same shape
    if ( !vector_same_shape(input0, input1) || !vector_same_shape(input0, input2) )
    {
        rCode = 1;
        PyErr_Format(PyExc_ValueError, "Shape mismatch");
    }

    if(!rCode) {
        N = PyArray_DIMS(input1)[0];

        // Ensure inputs are in 64 bit double format and contiguous and in FORTRAN order
        PyArray_Descr* npy_float64_descr = PyArray_DescrFromType(NPY_FLOAT64);

        L = (PyArrayObject*) PyArray_FromAny((PyObject*)input1, npy_float64_descr, 2, 2, NPY_ARRAY_F_CONTIGUOUS, NULL);
        F = (PyArrayObject*) PyArray_FromAny((PyObject*)input2, npy_float64_descr, 2, 2, NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_ENSURECOPY, NULL);
 
        if(L == NULL || F == NULL) {
            rCode = 2;
            PyErr_Format(PyExc_ValueError, "Could not allocate L or F storage");
        }
    }

    if(!rCode) {
    
        int F_str0 = PyArray_STRIDES(F)[0] / PyArray_ITEMSIZE(F);
        int F_str1 = PyArray_STRIDES(F)[1] / PyArray_ITEMSIZE(F);

        int L_str0 = PyArray_STRIDES(L)[0] / PyArray_ITEMSIZE(L);
        int L_str1 = PyArray_STRIDES(L)[1] / PyArray_ITEMSIZE(L);

        if(UPLO=='L'){
            tril((npy_float64*)PyArray_DATA(F), F_str0, F_str1, N);
        }
        else {
            triu((npy_float64*)PyArray_DATA(F), F_str0, F_str1, N);
        }

        //printMatrix("L",        (npy_float64*)PyArray_DATA(L), L_str0, L_str1, N);
        //printMatrix("F_before", (npy_float64*)PyArray_DATA(F), F_str0, F_str1, N);

        // Call the library function
        int info = choleskyGrad(UPLO, (double*)PyArray_DATA(L), (double*)PyArray_DATA(F), N);

        //printMatrix("F_after", (npy_float64*)PyArray_DATA(F), F_str0, F_str1, N);
    
        if(info != 0) {
            rCode = 3;
            Py_XDECREF(F);
            PyErr_Format(PyExc_ValueError, "cholgrad library returned error code %d\n", info);
        }
        // We don't need the reference to L anymore
        //Py_XDECREF(L); // Don't think we need this
    }

    if(!rCode) {
        // Check whether there was any memory already attached to the output
        if (*output0 != NULL) {
            // If so, decrement the count to that memory
            Py_XDECREF(*output0);
        }
	

        // Get descriptor of output type
        PyArray_Descr* output_descr =  PyArray_DescrFromType(TYPENUM_OUTPUT_0);
	    // Copy the F to the right type format for the output: Can use C or F memory styles
        *output0 = (PyArrayObject*) PyArray_FromAny((PyObject*)F, output_descr, 2, 2, NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_FORCECAST, NULL);
        if(*output0 == NULL) {
            rCode = 4;
            PyErr_Format(PyExc_ValueError, "Could not allocate output storage");
        }
        // Don't need the reference to F anymore
        //Py_XDECREF(F); // Don't think we need this
        Py_XDECREF(output_descr);
    }

    return rCode;
}
