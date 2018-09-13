#ifndef RESULT_H
#define RESULT_H

typedef struct resultFGMRES {
  int restart, iter, steps, restarts;
  double time, timeJacobiLower, timeJacobiUpper;
  double averageTimeJacobiLower, averageTimeJacobiUpper;
  double timeFgmres, averageTimeFgmres;
  double averageNormJacobiLower, averageNormJacobiUpper;

} resultFGMRES;

typedef struct resultJacobi {
  int sections;
  int *sectionsConf;
  int *sectionsIter;
  double averageTimeJacobiLower, averageTimeJacobiUpper;
  double averageNormJacobiLower, averageNormJacobiUpper;
  double maxNormJacobiLower, minNormJacobiLower;
  double maxNormJacobiUpper, minNormJacobiUpper;

} resultJacobi;

#endif //RESULT_H