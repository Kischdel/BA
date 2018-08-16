#ifndef JACOBI_PAR_H
#define JACOBI_PAR_H

void jacobi(SparseMatrix *I, const int iterations, double *b, double *x, void (*lower)(const int, const int, const int, double*, double*, int*, int*, double*));

void lowerLine(const int lineIndex, const int nBlock, const int n, double *b, double *x, int *row, int *col, double *val);
void upperLine(const int lineIndex, const int nBlock, const int n, double *b, double *x, int *row, int *col, double *val);

void jacobiLower(SparseMatrix* I, const int iterations, double *b, double *x);
void jacobiUpper(SparseMatrix* I, const int iterations, double *b, double *x);

void jacobiLowerAsync(SparseMatrix* I, const int iterations, double *b, double *x);
void jacobiUpperAsync(SparseMatrix* I, const int iterations, double *b, double *x);

#endif //JACOBI_PAR_H