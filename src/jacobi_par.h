#ifndef JACOBI_PAR_H
#define JACOBI_PAR_H

void jacobi(SparseMatrix *I, const int iterations, double *b, double *x, void (*lower)(const int, const int, const int, double*, double*, int*, int*, double*));

void lowerLine(const int lineIndex, const int nBlock, const int n, double *b, double *x, int *row, int *col, double *val);
void upperLine(const int lineIndex, const int nBlock, const int n, double *b, double *x, int *row, int *col, double *val);

void jacobiLowerPatternSync(SparseMatrix *I, const int iterations, double *b, double *x);
void jacobiUpperPatternSync(SparseMatrix *I, const int iterations, double *b, double *x);

void jacobiLowerSync(SparseMatrix *I, const int iterations, double *b, double *x0, double *x);
void jacobiUpperSync(SparseMatrix *I, const int iterations, double *b, double *x0, double *x);

void jacobiLowerHalfSync(SparseMatrix *I, const int iterations, double *b, double *x);
void jacobiUpperHalfSync(SparseMatrix *I, const int iterations, double *b, double *x);

void jacobiLowerAsync(SparseMatrix *I, const int iterations, double *b, double *x);
void jacobiUpperAsync(SparseMatrix *I, const int iterations, double *b, double *x);

double computeResidualLower(SparseMatrix *I, double *b, double *x, double *residual);
double computeResidualUpper(SparseMatrix *I, double *b, double *x, double *residual);

void simpleJacobi(SparseMatrix* I, const int iterations, double *b, double *x0, double *x);
void simpleJacobiLower(SparseMatrix* I, const int iterations, double *b, double *x0, double *x);
void simpleJacobiUpper(SparseMatrix* I, const int iterations, double *b, double *x0, double *x);

double simpleResidual(SparseMatrix *I, double *b, double *x, double *residual);
double simpleResidualLower(SparseMatrix *I, double *b, double *x, double *residual);
double simpleResidualUpper(SparseMatrix *I, double *b, double *x, double *residual);

#endif //JACOBI_PAR_H