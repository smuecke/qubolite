#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>


PyObject *brute_force_(PyObject *self, PyObject *args) {
    PyArrayObject *arr;
    PyArg_ParseTuple(args, "O", &arr);
    if (PyErr_Occurred() || !PyArray_Check(arr))
	    return NULL;

    size_t n = PyArray_DIM(arr, 0);
    double **qubo;
    npy_intp dims[] = { [0] = n, [1] = n };
    PyArray_AsCArray((PyObject **)&arr, &qubo, dims, 2,
        PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred()) return NULL;

    double x[n]; /* bit vector */
    double dxi;
    double val = 0; /* QUBO value */

    double min_x[n];
    double min_val = 0;

    int64_t it = 0; /* total iteration counter */
    int64_t it_;
    size_t i = 0;   /* bit flip index */
    size_t j;

    /* init x and min_x */
    for (j=0; j<n; ++j) {
        x[j] = 0.0;
        min_x[j] = 0.0;
    }

    while (it < (1 << n)-1) {
        /* get next bit flip index (gray code) */
        it_ = it;
        i = 0;
        while (it_ & 1) {
            ++i;
            it_ = it_ >> 1;
        }

        x[i] = 1.0 - x[i];
        dxi = 2.0*x[i]-1.0;
        /* calculate function value offset */
        for (j = 0; j < i; ++j) {
            if (x[j] <= 0)
                continue;
            val += dxi * qubo[j][i];
        }
        val += dxi * qubo[i][i];
        for (j = i+1; j < n; ++j) {
            if (x[j] <= 0)
                continue;
            val += dxi * qubo[i][j];
        }

        if (val < min_val) {
            min_val = val;
            /* copy array */
            for (j = 0; j < n; ++j)
                min_x[j] = x[j];
        }
        ++it;
    }
    return PyFloat_FromDouble(min_val);
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
