#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/random/bitgen.h>
#include <numpy/random/distributions.h>
#include <omp.h>

typedef unsigned char bit;

void print_bits(bit *x, size_t n) {
    for (size_t i=0; i<n; ++i)
        printf("%d ", x[i]);
    printf("\n");
}

double qubo_score(double **qubo, bit *x, const size_t n) {
    double v = 0.0;
    for (size_t i=0; i<n; ++i) {
        if (x[i]<=0)
            continue;
        v += qubo[i][i];
        for (size_t j=i+1; j<n; ++j)
            v += x[j]*qubo[i][j];
    }
    return v;
}

double qubo_score_condition_1(double **qubo, bit *x, const size_t n, const size_t i) {
    double v = qubo[i][i];
    size_t j=0;
    for (; j<i; ++j)
        v += x[j] * qubo[j][i];
    for (j=i+1; j<n; ++j)
        v += x[j] * qubo[i][j];
    return v;
}

struct _brute_force_result {
    bit *min_x;
    double min_val0;
    double min_val1;
} typedef brute_force_result;

brute_force_result _brute_force(
        double **qubo,
        const size_t n,
        size_t n_fixed_bits) {
    bit *x = (bit*)malloc(n*sizeof(bit)); // bit vector
    memset(x, 0, n);

    // fix some bits
#ifndef __APPLE__
    const size_t thread_id = omp_get_thread_num();
#else
    const size_t thread_id = 0;
#endif
    for (size_t k=0; k<n_fixed_bits; ++k)
        x[n-k-1] = (thread_id & (1ULL<<k))>0 ? 1 : 0;
    double val = qubo_score(qubo, x, n); // QUBO value
    double dval; // QUBO value update

    bit *min_x = (bit*) malloc(n);
    memcpy(min_x, x, n);
    double min_vals[2] = {val, INFINITY};

    size_t i, j;
    // make sure it_lim is correctly set
    int64_t it_lim = (1ULL<<n-n_fixed_bits)-1;
    for (int64_t it=0; it<it_lim; ++it) {
        // get next bit flip index (gray code)
#ifdef _MSC_VER
        i = _tzcnt_u64(~it);
#else
        i = __builtin_ctzll(~it);
#endif
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
    free(x);
    return res;
}

PyObject *py_brute_force(PyObject *self, PyObject *args) {
#ifndef __APPLE__
    const size_t MAX_THREADS = omp_get_max_threads();
#else
    const size_t MAX_THREADS = 1;
#endif
    PyArrayObject *arr;
    size_t max_threads = MAX_THREADS;
    PyArg_ParseTuple(args, "O|k", &arr, &max_threads);
    if (PyErr_Occurred() || !PyArray_Check(arr))
	    return NULL;

    const size_t n = PyArray_DIM(arr, 0);
    double **qubo;
    npy_intp dims[] = { [0] = n, [1] = n };
    PyArray_AsCArray((PyObject**) &arr, &qubo, dims, 2,
        PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred())
        return NULL;

#ifdef _MSC_VER
    size_t m = 63-__lzcnt64(max_threads); // floor(log2(MAX_THREADS))
#else
    size_t m = 63-__builtin_clzll(max_threads); // floor(log2(MAX_THREADS))
#endif

    // ensure that the number of bits to optimize is positive
    if (n<=m) m = n-1;

    // check if n is too large and would cause an
    // overflow of size64_t
    if (n-m>=64) {
    	// return None
	    Py_INCREF(Py_None);
	    return Py_None;
    }

    const size_t M = 1ULL<<m; // first power of 2 less or equals MAX_THREADS
#ifndef __APPLE__
    omp_set_dynamic(0);
#endif
    brute_force_result *ress = (brute_force_result*)malloc(M*sizeof(brute_force_result));
    #pragma omp parallel num_threads(M)
    {
#ifndef __APPLE__
        ress[omp_get_thread_num()] = _brute_force(qubo, n, m);
#else
        ress[0] = _brute_force(qubo, n, m);
#endif
    }

    // collect all min values (except first result)
    double *all_vals = (double*)malloc((2*M-2)*sizeof(double));
    for (size_t j=1; j<M; ++j) {
        all_vals[2*j-2] = ress[j].min_val0;
        all_vals[2*j-1] = ress[j].min_val1;
    }

    // sort again to get two lowest values
    double global_min_val0 = ress[0].min_val0;
    double global_min_val1 = ress[0].min_val1;
    size_t global_min_ix = 0;
    for (size_t j=0; j<2*M-2; ++j) {
        if (all_vals[j]<global_min_val1) {
            if (all_vals[j]<global_min_val0) {
                global_min_val1 = global_min_val0;
                global_min_val0 = all_vals[j];
                global_min_ix = (j>>1)+1;
            } else if (all_vals[j]>global_min_val0) {
                global_min_val1 = all_vals[j];
            }
        }
    }

    // prepare return values
    PyObject *min_x_obj = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
    double *min_x_obj_data = PyArray_DATA((PyArrayObject*) min_x_obj);
    bit *global_min_x = ress[global_min_ix].min_x;
    for (size_t j=0; j<n; ++j)
        min_x_obj_data[j] = (double) global_min_x[j];
    for (size_t j=0; j<M; ++j)
        free(ress[j].min_x);
    free(ress);
    PyObject *min_val0_obj = PyFloat_FromDouble(global_min_val0);
    PyObject *min_val1_obj = PyFloat_FromDouble(global_min_val1);
    PyObject *tup = PyTuple_New(3);
    PyTuple_SetItem(tup, 0, min_x_obj);
    PyTuple_SetItem(tup, 1, min_val0_obj);
    PyTuple_SetItem(tup, 2, min_val1_obj);
    if (PyErr_Occurred())
        return NULL;
    return tup;
}

/* ################################################
 * Gibbs sampling
 * ################################################ */

void _gibbs_sample(const size_t n, double **qubo, bit *state, const size_t rounds, bitgen_t *random_engine) {
    double p, u;
    for (size_t i=0; i<rounds; ++i) {
        for (size_t v=0; v<n; ++v) {
            p = exp(-qubo_score_condition_1(qubo, state, n, v));
            u = (double) random_uniform(random_engine, 0.0, p+1.0);
            if ( !(!state[v] ^ (u < p)) )
                state[v] = !state[v];
        }
    }
}

PyObject *py_gibbs_sample(PyObject *self, PyObject *args) {
#ifndef __APPLE__
    const size_t MAX_THREADS = omp_get_max_threads();
#else
    const size_t MAX_THREADS = 1;
#endif
    PyArrayObject *arr;
    size_t max_threads = 1;
    size_t num_samples = 1;
    size_t burn = 100;
    size_t keep = 100;
    PyObject *bitgencaps = Py_None;
    PyArg_ParseTuple(args, "OO|kkkk", &arr, &bitgencaps, &num_samples, &burn, &max_threads, &keep);
    if (PyErr_Occurred() || !PyArray_Check(arr))
            return NULL;

    max_threads = (num_samples < max_threads) ? num_samples : max_threads;
    max_threads = (MAX_THREADS < max_threads) ? MAX_THREADS : max_threads;

    bitgen_t *random_engine[max_threads];
    for (size_t i=0; i<max_threads; ++i)
        random_engine[i] = (bitgen_t*) PyCapsule_GetPointer(PyList_GET_ITEM(bitgencaps, i), "BitGenerator");

    const size_t n = PyArray_DIM(arr, 0);
    double **qubo;
    npy_intp dims[] = { [0] = n, [1] = n };
    PyArray_AsCArray((PyObject**) &arr, &qubo, dims, 2,
        PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred())
        return NULL;

    npy_intp rdims[] = { [0] = num_samples, [1] = n };
    PyObject *res = PyArray_SimpleNew(2, rdims, NPY_UINT8);
    Py_INCREF(res);
    bit *samples = (bit*)PyArray_DATA((PyArrayObject*)res);

    bit *chain_state = (bit*)malloc(sizeof(bit)*max_threads*n);
    for (size_t i=0; i<max_threads*n; ++i)
        chain_state[i] = (bit) (random_uint(*random_engine) % 2);

#ifndef __APPLE__
    omp_set_dynamic(0);
#endif
#pragma omp parallel for num_threads(max_threads)
    for (size_t j=0; j<num_samples; ++j) {
#ifndef __APPLE__
        const size_t tid = omp_get_thread_num();
#else
        const size_t tid = 0;
#endif
        bit *tstate = chain_state+(tid*n);
	_gibbs_sample(n, qubo, tstate, j==tid ? burn : keep, random_engine[tid]);
        memcpy(samples+(j*n), tstate, sizeof(bit)*n);
    }

    free(chain_state);
    return res;
}


/* ################################################
 * Annealing
 * ################################################ */

double beta_ip1(double b) {
    return ( 1.0 + sqrt(4.0*b*b+1.0) ) / 2.0;
}

void _anneal(const size_t n, double **qubo, bit *state, const size_t rounds, bitgen_t *random_engine) {
    double p, u, bi = 1;
    for (size_t i=0; i<rounds; ++i) {
        for (size_t v=0; v<n; ++v) {
            p = exp(-qubo_score_condition_1(qubo, state, n, v) * bi);
            u = (double) random_uniform(random_engine, 0.0, p+1.0);
            if ( !(!state[v] ^ (u < p)) )
                state[v] = !state[v];
        }
        bi = beta_ip1(bi);
    }
}

PyObject *py_anneal(PyObject *self, PyObject *args) {
#ifndef __APPLE__
    const size_t MAX_THREADS = omp_get_max_threads();
#else
    const size_t MAX_THREADS = 1;
#endif
    PyArrayObject *arr;
    size_t max_threads = 1;
    size_t num_samples = 1;
    size_t burn = 100;
    size_t keep = 100;
    PyObject *bitgencaps = Py_None;
    PyArg_ParseTuple(args, "OO|kkkk", &arr, &bitgencaps, &num_samples, &burn, &max_threads, &keep);
    if (PyErr_Occurred() || !PyArray_Check(arr))
            return NULL;

    max_threads = (num_samples < max_threads) ? num_samples : max_threads;
    max_threads = (MAX_THREADS < max_threads) ? MAX_THREADS : max_threads;

    bitgen_t *random_engine[max_threads];
    for (size_t i=0; i<max_threads; ++i)
        random_engine[i] = (bitgen_t*) PyCapsule_GetPointer(PyList_GET_ITEM(bitgencaps, i), "BitGenerator");

    const size_t n = PyArray_DIM(arr, 0);
    double **qubo;
    npy_intp dims[] = { [0] = n, [1] = n };
    PyArray_AsCArray((PyObject**) &arr, &qubo, dims, 2,
        PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred())
        return NULL;

    npy_intp rdims[] = { [0] = num_samples, [1] = n };
    PyObject *res = PyArray_SimpleNew(2, rdims, NPY_UINT8);
    Py_INCREF(res);
    bit *samples = (bit*)PyArray_DATA((PyArrayObject*)res);

    bit *chain_state = (bit*)malloc(sizeof(bit)*max_threads*n);
    for (size_t i=0; i<max_threads*n; ++i)
        chain_state[i] = (bit) (random_uint(*random_engine) % 2);

#ifndef __APPLE__
    omp_set_dynamic(0);
#endif
#pragma omp parallel for num_threads(max_threads)
    for (size_t j=0; j<num_samples; ++j) {
#ifndef __APPLE__
        const size_t tid = omp_get_thread_num();
#else
        const size_t tid = 0;
#endif
        bit *tstate = chain_state+(tid*n);
	_anneal(n, qubo, tstate, j==tid ? burn : keep, random_engine[tid]);
        memcpy(samples+(j*n), tstate, sizeof(bit)*n);
    }

    free(chain_state);
    return res;
}


/* ################################################
 * Python module def                              
 * ################################################ */

static PyMethodDef methods[] = {
    {"brute_force", py_brute_force, METH_VARARGS, "Solves QUBO the hard way"},
    {"gibbs_sample", py_gibbs_sample, METH_VARARGS, "Sample from the induced exponential family"},
    {"anneal", py_anneal, METH_VARARGS, "Experimental QUBO solver, based on magic annealing schedule"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_c_utils",
    NULL, -1, methods
};

PyMODINIT_FUNC PyInit__c_utils() {
    import_array();
    return PyModule_Create(&module);
}
