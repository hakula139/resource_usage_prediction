#include "utils.hpp"

#include <algorithm>
#include <string>
#include <tuple>

#include "config.hpp"
#include "sciplot/sciplot.hpp"
#include "types.hpp"

sciplot::Plot NewPlot(size_t size) {
  sciplot::Plot plot;

  auto figure_size = [](size_t data_size) -> std::tuple<size_t, size_t> {
    return {std::max(data_size << 2, 1920ul), 920ul};
  };
  auto [width, height] = figure_size(size);

  plot.size(width, height);
  plot.fontName("Palatino");
  plot.fontSize(16);
  plot.legend().atTop().fontSize(16).displayHorizontal();
  plot.palette("gnpu");

  return plot;
}

void PlotPredictions(
    const std::string& figure_path,
    const std::vector<int64_t>& x1,
    const std::vector<value_t>& y1,
    const std::vector<int64_t>& x2,
    const std::vector<value_t>& y2) {
  auto plot = NewPlot(x1.size());
  plot.xlabel("Epoch");
  plot.ylabel("Value");
  plot.drawCurve(x1, y1).label("Expected");
  plot.drawCurve(x2, y2).label("Predictions");
  plot.save(figure_path);
}

void PlotLoss(
    const std::string& figure_path,
    const std::vector<int64_t>& x1,
    const std::vector<double>& y1,
    const std::vector<int64_t>& x2,
    const std::vector<double>& y2) {
  auto plot = NewPlot(x1.size());
  plot.xlabel("Epoch");
  plot.ylabel("Loss");
  plot.drawCurve(x1, y1).label("Training loss");
  plot.drawCurve(x2, y2).label("Validation loss (average)");
  plot.save(figure_path);
}
