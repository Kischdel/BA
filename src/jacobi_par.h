#ifndef JACOBI_PAR_H
#define JACOBI_PAR_H

// blockasynchron jacobi with busy wait barrier
void jacobiDynBusy(SparseMatrix *I, double *b, double *x, void (*func)(const int, const int, const int, double*, double*, int*, int*, double*), int blockCount, int iterations);

// blockasynchron jacobi with nested parallel regions
void jacobi(SparseMatrix *I, double *b, double *x, void (*func)(const int, const int, const int, double*, double*, int*, int*, double*), int secNum, int *secConf, int *secIter);

// functions to update a single line
void lowerLine(const int lineIndex, const int nBlock, const int n, double *b, double *x, int *row, int *col, double *val);
void upperLine(const int lineIndex, const int nBlock, const int n, double *b, double *x, int *row, int *col, double *val);

// Default Jacobi
void jacobiLowerSync(SparseMatrix *I, const int iterations, double *b, double *x0, double *x);
void jacobiUpperSync(SparseMatrix *I, const int iterations, double *b, double *x0, double *x);

// Concurrent jacobi
void jacobiLowerHalfSync(SparseMatrix *I, const int iterations, double *b, double *x);
void jacobiUpperHalfSync(SparseMatrix *I, const int iterations, double *b, double *x);

// Asyncron jacobi
void jacobiLowerAsync(SparseMatrix *I, const int iterations, double *b, double *x);
void jacobiUpperAsync(SparseMatrix *I, const int iterations, double *b, double *x);

// residual computation
double computeResidualLower(SparseMatrix *I, double *b, double *x, double *residual);
double computeResidualUpper(SparseMatrix *I, double *b, double *x, double *residual);

// function that doesn't do anything, to determine the Overhead of parallelization
void parallelOverheadLower(SparseMatrix *I, const int iterations, double *b, double *x);
void parallelOverheadUpper(SparseMatrix *I, const int iterations, double *b, double *x);

// simple algorithms are not optimized for matrix structure and use naive approach. They are used to test correctness
void simpleJacobi(SparseMatrix* I, const int iterations, double *b, double *x0, double *x);
void simpleJacobiLower(SparseMatrix* I, const int iterations, double *b, double *x0, double *x);
void simpleJacobiUpper(SparseMatrix* I, const int iterations, double *b, double *x0, double *x);

double simpleResidual(SparseMatrix *I, double *b, double *x, double *residual);
double simpleResidualLower(SparseMatrix *I, double *b, double *x, double *residual);
double simpleResidualUpper(SparseMatrix *I, double *b, double *x, double *residual);

#endif //JACOBI_PAR_H