#include "utils.hpp"

#include <sciplot/sciplot.hpp>

#include "config.hpp"

void PlotPredictions(
    const std::vector<int64_t>& x1,
    const std::vector<int64_t>& y1,
    const std::vector<int64_t>& x2,
    const std::vector<int64_t>& y2) {
  sciplot::Plot plot;

  plot.size(1000, 600);
  plot.fontName("Palatino");
  plot.fontSize(16);
  plot.xlabel("Epoch");
  plot.ylabel("Count");
  plot.legend().atTop().fontSize(16).displayHorizontal();
  plot.palette("gnpu");

  plot.drawCurve(x1, y1).label("Expected");
  plot.drawCurve(x2, y2).label("Predictions");

  plot.save(PREDICTIONS_FIGURE_PATH);
}

void PlotLoss(const std::vector<int64_t>& x, const std::vector<int64_t>& y) {
  sciplot::Plot plot;

  plot.size(1000, 600);
  plot.fontName("Palatino");
  plot.fontSize(16);
  plot.xlabel("Epoch");
  plot.ylabel("Loss");
  plot.legend().atTop().fontSize(16).displayHorizontal();

  plot.drawCurve(x, y).label("Training loss");

  plot.save(LOSS_FIGURE_PATH);
}
