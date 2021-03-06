#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include <vector> 

class SparseMatrix {
  private:
    typedef struct entry {
      int i, j;
      double val;
  
      bool operator < (const entry& str) const
        {
          if(i < str.i) return 1;
          if(i > str.i) return 0;
          if(j < str.j) return 1;
          return 0;
        }
    
    } ent;
  
    static void calculateEntrys(int nBlock, std::vector<ent*> *list);
    
    static bool PComp(const ent * const & a, const ent * const & b) { return *a < *b; }
  

  public:
  
  int n;
  int nBlock;
  
  double *val;
  int valSize;
  int *col;
  int *row;
  int rowSize;
  
  
  
  
  SparseMatrix(int nBlock);
  ~SparseMatrix();

};

#endif //SPARSEMATRIX_H