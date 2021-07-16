#include "utils.hpp"

#include <cmath>

double NormalPdf(double x, double mean, double sd) {
  auto sd_2 = pow(sd, 2);
  return exp(-pow(x - mean, 2) / (2 * sd_2)) / sqrt(2 * M_PI * sd_2);
}
