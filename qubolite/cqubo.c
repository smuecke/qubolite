#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <omp.h>

typedef unsigned char bit;


double qubo_score(double **qubo, bit *x, size_t n) {
    double v = 0.0;
    for (size_t i=0; i<n; ++i) {
        if (x[i]<=0)
            continue;
        v += qubo[i][i];
        for (size_t j=i+1; j<n; ++j) {
            v += x[j]*qubo[i][j];
        }
    }
    return v;
}

struct _brute_force_result {
    bit *min_x;
    double min_val0;
    double min_val1;
} typedef brute_force_result;

brute_force_result _brute_force(double **qubo, size_t n, size_t n_fixed_bits) {
    bit x[n]; // bit vector
    memset(x, 0, n);

    // fix some bits
    const size_t thread_id = omp_get_thread_num();
    for (size_t k=0; k<n_fixed_bits; ++k)
        x[n-k-1] = (thread_id & (1<<k))>0 ? 1 : 0;
    double val = qubo_score(qubo, x, n); // QUBO value
    double dval; // QUBO value update

    bit *min_x = (bit*) malloc(n);
    memset(min_x, 0, n);
    double min_vals[2];

    size_t i = 0; // bit flip index
    size_t j;
    for (int64_t it=0; it<(1<<(n-n_fixed_bits))-1; ++it) {
        // get next bit flip index (gray code)
        i = __builtin_ctzll(~it);

        x[i] ^= 1; // flip bit
        // calculate function value offset
        dval = 0;
        for (j=0; j<i; ++j)
            dval += x[j]*qubo[j][i];
        dval += qubo[i][i];
        for (j=i+1; j<n; ++j)
            dval += x[j]*qubo[i][j];

        // add or subtract, depending on bit change
        val += x[i] ? dval : -dval;
        // memorize two lowest values and bit vector
        // with lowest value (minimum solution)
        if (val<min_vals[1])
            if (val<min_vals[0]) {
                min_vals[1] = min_vals[0];
                min_vals[0] = val;
                memcpy(min_x, x, n);
            } else if (val>min_vals[0])
                min_vals[1] = val;
    }

    brute_force_result res = {min_x, min_vals[0], min_vals[1]};
    return res;
}

PyObject *py_brute_force(PyObject *self, PyObject *args) {
    PyArrayObject *arr;
    PyArg_ParseTuple(args, "O", &arr);
    if (PyErr_Occurred() || !PyArray_Check(arr))
	    return NULL;

    const size_t n = PyArray_DIM(arr, 0);
    double **qubo;
    npy_intp dims[] = { [0] = n, [1] = n };
    PyArray_AsCArray((PyObject**) &arr, &qubo, dims, 2,
        PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred())
        return NULL;

    const size_t MAX_THREADS = omp_get_max_threads();
    size_t m = 63-__builtin_clzll(MAX_THREADS);
    // ensure that the number of bits to optimize is positive
    if (n<=m) m = n-1;
    const size_t M = 1<<m; // first power of 2 less or equals MAX_THREADS
    omp_set_dynamic(0);
    brute_force_result ress[M];
    #pragma omp parallel num_threads(M)
    {
        ress[omp_get_thread_num()] = _brute_force(qubo, n, m);
    }

    brute_force_result res = ress[0];
    for (size_t j=0; j<M; ++j) {
        if (ress[j].min_val0<res.min_val0)
            res = ress[j];
    }

    // prepare return values
    PyObject *min_x_obj = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
    double *min_x_obj_data = PyArray_DATA((PyArrayObject*) min_x_obj);
    for (size_t j=0; j<n; ++j)
        min_x_obj_data[j] = (double) res.min_x[j];
    for (size_t j=0; j<M; ++j)
        free(ress[j].min_x);
    PyObject *min_val0_obj = PyFloat_FromDouble(res.min_val0);
    PyObject *min_val1_obj = PyFloat_FromDouble(res.min_val1);
    PyObject *tup = PyTuple_New(3);
    PyTuple_SetItem(tup, 0, min_x_obj);
    PyTuple_SetItem(tup, 1, min_val0_obj);
    PyTuple_SetItem(tup, 2, min_val1_obj);
    if (PyErr_Occurred())
        return NULL;
    return tup;
}

static PyMethodDef methods[] = {
    {"brute_force", py_brute_force, METH_VARARGS, "Solves QUBO the hard way"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cqubo = {
    PyModuleDef_HEAD_INIT, "cqubo",
    NULL, -1, methods
};

PyMODINIT_FUNC PyInit_cqubo() {
    import_array();
    return PyModule_Create(&cqubo);
}
