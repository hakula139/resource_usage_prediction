#include "utils.hpp"

#include <sciplot/sciplot.hpp>

#include "config.hpp"

sciplot::Plot NewPlot(int64_t size) {
  sciplot::Plot plot;

  plot.size(size * 4, 920);
  plot.fontName("Palatino");
  plot.fontSize(16);
  plot.legend().atTop().fontSize(16).displayHorizontal();
  plot.palette("gnpu");

  return plot;
}

void PlotPredictions(
    const std::vector<int64_t>& x1,
    const std::vector<int64_t>& y1,
    const std::vector<int64_t>& x2,
    const std::vector<int64_t>& y2) {
  auto plot = NewPlot(x1.size());
  plot.xlabel("Epoch");
  plot.ylabel("Count");
  plot.drawCurve(x1, y1).label("Expected");
  plot.drawCurve(x2, y2).label("Predictions");
  plot.save(PREDICTIONS_FIGURE_PATH);
}

void PlotTrainLoss(
    const std::vector<int64_t>& x1,
    const std::vector<double>& y1,
    const std::vector<int64_t>& x2,
    const std::vector<double>& y2) {
  auto plot = NewPlot(x1.size());
  plot.xlabel("Epoch");
  plot.ylabel("Loss");
  plot.drawCurve(x1, y1).label("Training loss");
  plot.drawCurve(x2, y2).label("Naive method loss");
  plot.save(TRAIN_LOSS_FIGURE_PATH);
}
