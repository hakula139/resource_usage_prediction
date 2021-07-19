#include "utils.hpp"

#include <sciplot/sciplot.hpp>

#include "config.hpp"

sciplot::Plot NewPlot(int64_t width, int64_t height = 900) {
  sciplot::Plot plot;

  plot.size(width, height);
  plot.fontName("Palatino");
  plot.fontSize(16);
  plot.xlabel("Epoch");
  plot.ylabel("Count");
  plot.legend().atTop().fontSize(16).displayHorizontal();
  plot.palette("gnpu");

  return plot;
}

void PlotPredictions(
    const std::vector<int64_t>& x1,
    const std::vector<int64_t>& y1,
    const std::vector<int64_t>& x2,
    const std::vector<int64_t>& y2) {
  auto plot = NewPlot(x1.size() * 3);
  plot.drawCurve(x1, y1).label("Expected");
  plot.drawCurve(x2, y2).label("Predictions");
  plot.save(PREDICTIONS_FIGURE_PATH);
}

void PlotTrainLoss(
    const std::vector<int64_t>& x1,
    const std::vector<double>& y1,
    const std::vector<int64_t>& x2,
    const std::vector<double>& y2) {
  auto plot = NewPlot(x1.size() * 3);
  plot.drawCurve(x1, y1).label("Training loss");
  plot.drawCurve(x2, y2).label("Naive method loss");
  plot.save(TRAIN_LOSS_FIGURE_PATH);
}

void PlotValidLoss(
    const std::vector<int64_t>& x1,
    const std::vector<double>& y1,
    const std::vector<int64_t>& x2,
    const std::vector<double>& y2) {
  auto plot = NewPlot(x1.size() * 3);
  plot.drawCurve(x1, y1).label("Validation loss");
  plot.drawCurve(x2, y2).label("Naive method loss");
  plot.save(VALID_LOSS_FIGURE_PATH);
}
