#ifndef JACOBI_PAR_H
#define JACOBI_PAR_H

void jacobiDynBusy(SparseMatrix *I, double *b, double *x, void (*func)(const int, const int, const int, double*, double*, int*, int*, double*), int blockCount, int iterations);

void jacobi(SparseMatrix *I, double *b, double *x, void (*func)(const int, const int, const int, double*, double*, int*, int*, double*), int secNum, int *secConf, int *secIter);

void lowerLine(const int lineIndex, const int nBlock, const int n, double *b, double *x, int *row, int *col, double *val);
void upperLine(const int lineIndex, const int nBlock, const int n, double *b, double *x, int *row, int *col, double *val);

void jacobiLowerSync(SparseMatrix *I, const int iterations, double *b, double *x0, double *x);
void jacobiUpperSync(SparseMatrix *I, const int iterations, double *b, double *x0, double *x);

void jacobiLowerHalfSync(SparseMatrix *I, const int iterations, double *b, double *x);
void jacobiUpperHalfSync(SparseMatrix *I, const int iterations, double *b, double *x);

void jacobiLowerAsync(SparseMatrix *I, const int iterations, double *b, double *x);
void jacobiUpperAsync(SparseMatrix *I, const int iterations, double *b, double *x);

double computeResidualLower(SparseMatrix *I, double *b, double *x, double *residual);
double computeResidualUpper(SparseMatrix *I, double *b, double *x, double *residual);

// Version that doesn't do anything, to determine the Overhead of parallelization
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