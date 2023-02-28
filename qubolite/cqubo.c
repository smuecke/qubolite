#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

typedef unsigned char bit;


PyObject *brute_force_(PyObject *self, PyObject *args) {
    PyArrayObject *arr;
    PyArg_ParseTuple(args, "O", &arr);
    if (PyErr_Occurred() || !PyArray_Check(arr))
	    return NULL;

    size_t n = PyArray_DIM(arr, 0);
    double **qubo;
    npy_intp dims[] = { [0] = n, [1] = n };
    PyArray_AsCArray((PyObject**) &arr, &qubo, dims, 2,
        PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred())
        return NULL;

    bit x[n]; // bit vector
    double val = 0; // QUBO value
    double dval; // QUBO value update

    bit min_x[n];
    double min_vals[2];

    int64_t it = 0; // total iteration counter
    int64_t it_;
    size_t i = 0; // bit flip index
    size_t j;

    // init x and min_x
    for (j=0; j<n; ++j) {
        x[j] = 0;
        min_x[j] = 0;
    }

    while (it<(1<<n)-1) {
        // get next bit flip index (gray code)
        it_ = it;
        i = 0;
        while (it_&1) {
            ++i;
            it_ = it_>>1;
        }

        x[i] ^= 1; // flip bit
        // calculate function value offset
        dval = 0;
        for (j=0; j<i; ++j) {
            if (x[j]>0)
                dval += qubo[j][i];
        }
        dval += qubo[i][i];
        for (j=i+1; j<n; ++j) {
            if (x[j]>0)
                dval += qubo[i][j];
        }

        // add or subtract, depending on bit change
        val += x[i] ? dval : -dval;
        // memorize two lowest values and bit vector
        // with lowest value (minimum solution)
        if (val<min_vals[1])
            if (val<min_vals[0]) {
                min_vals[1] = min_vals[0];
                min_vals[0] = val;
                // copy array
                for (j=0; j<n; ++j)
                    min_x[j] = x[j];
            } else if (val>min_vals[0])
                min_vals[1] = val;
        ++it;
    }

    // prepare return values
    PyObject *min_x_obj = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
    double *min_x_obj_data = PyArray_DATA((PyArrayObject*) min_x_obj);
    for (j=0; j<n; ++j)
        min_x_obj_data[j] = (double) min_x[j];
    PyObject *min_val0_obj = PyFloat_FromDouble(min_vals[0]);
    PyObject *min_val1_obj = PyFloat_FromDouble(min_vals[1]);
    PyObject *res = PyTuple_New(3);
    PyTuple_SetItem(res, 0, min_x_obj);
    PyTuple_SetItem(res, 1, min_val0_obj);
    PyTuple_SetItem(res, 2, min_val1_obj);
    if (PyErr_Occurred())
        return NULL;
    return res;
}

static PyMethodDef methods[] = {
    {"brute_force", brute_force_, METH_VARARGS, "Solves QUBO the hard way"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cqubo = {
    PyModuleDef_HEAD_INIT, "cqubo",
    NULL, -1, methods
};

PyMODINIT_FUNC PyInit_cqubo() {
    printf("Initialize module cqubo\n");
    import_array();
    return PyModule_Create(&cqubo);
}
