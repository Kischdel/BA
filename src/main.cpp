#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include "parex.h"
#include "fgmres.h"
#include "sparsematrix.h"
#include "jacobi_par.h"



int main(int argc, char *argv[]) {
  
  int nBlock = 4;
  
  SparseMatrix *a = new SparseMatrix(nBlock);
  int n = a->n;
  double b[n];
  double tol = 1.0e-03;
  
  
  // compute b
  int counter = 0;
  
  for(int k = 1; k <= nBlock; k++) {
    for(int m = 1; m <= nBlock; m++ ) {
      
      double x = (double)m / (double)(nBlock + 1);
      double y = (double)k / (double)(nBlock + 1);
      
      b[counter++] = 32 * ( x * (1.0 - x) + y * (1.0 - y));
    }
  }
  
  
  // initialize x0
  double x[n];
  for(int i = 0; i < n; i++) {
    // x[i] = b[i];
    x[i] = 0;
  }
  
  
  
  
  
  // EXPERIMENTAL SECTION
  
  a->computeILU();
  a->saveToFile();
  
  
  /*
  
  // jacobiLower(&a, 1, b, x);
  
  a.printILU();
  a.computeILU();
  
  std::cout << "\n";
  
  a.printILU();
  
  std::cout << "\n";
  
  a.print();
    
    
  */
  
  //fgmres(nBlock, b, x, a.val, a.col, a.row, tol);
  
  
  /*
   std::cout << "solution:\n";
  for(int i = 0; i < n; i++) {
    std::cout << x[i] << "\n";
  }
  */
  
  
  // a.print();
  
  /*
  SparseMatrix *l;
  SparseMatrix *u;
  a.ILU(l, u);
  */
  
  return 0;
}