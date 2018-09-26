#ifndef RESULT_H
#define RESULT_H

#include <string>

// holds data of a FGMRES execution
typedef struct resultFGMRES {
  int restart, steps, restarts;
  double time, timeJacobiLower, timeJacobiUpper;
  double averageTimeJacobiLower, averageTimeJacobiUpper;
  double timeFgmres, averageTimeFgmres;
  double averageNormJacobiLower, averageNormJacobiUpper;

} resultFGMRES;

// holds data of a Jacobi execution
typedef struct resultJacobi {
  double averageTimeJacobiLower, averageTimeJacobiUpper;
  double averageNormJacobiLower, averageNormJacobiUpper;
  double maxNormJacobiLower, minNormJacobiLower;
  double maxNormJacobiUpper, minNormJacobiUpper;

} resultJacobi;

#endif //RESULT_H