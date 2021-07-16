#include "utils.hpp"

#include <sciplot/sciplot.hpp>

#include "config.hpp"

void Plot(
    const std::vector<int64_t>& x1,
    const std::vector<int64_t>& y1,
    const std::vector<int64_t>& x2,
    const std::vector<int64_t>& y2) {
  sciplot::Plot plot;

  plot.size(1000, 600);
  plot.fontName("Palatino");
  plot.fontSize(16);
  plot.xlabel("Time");
  plot.ylabel("Count");
  plot.legend().atTop().fontSize(16).displayHorizontal();

  plot.drawCurve(x1, y1).label("Expected");
  plot.drawCurve(x2, y2).label("Predictions");

  plot.save(PLOT_PATH);
}
