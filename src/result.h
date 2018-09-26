#ifndef RESULT_H
#define RESULT_H

#include <string>

typedef struct resultFGMRES {
  int restart, steps, restarts;
  double time, timeJacobiLower, timeJacobiUpper;
  double averageTimeJacobiLower, averageTimeJacobiUpper;
  double timeFgmres, averageTimeFgmres;
  double averageNormJacobiLower, averageNormJacobiUpper;

} resultFGMRES;

typedef struct resultJacobi {
  double averageTimeJacobiLower, averageTimeJacobiUpper;
  double averageNormJacobiLower, averageNormJacobiUpper;
  double maxNormJacobiLower, minNormJacobiLower;
  double maxNormJacobiUpper, minNormJacobiUpper;

} resultJacobi;

#endif //RESULT_H