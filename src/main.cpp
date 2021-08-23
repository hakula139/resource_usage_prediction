#include <chrono>  // NOLINT(build/c++11)
#include <fstream>
#include <future>  // NOLINT(build/c++11)
#include <iostream>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <unordered_map>

#include "common/config.hpp"
#include "common/types.hpp"
#include "worker.hpp"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

// Feed the latest data and get the prediction for the next value
std::future<value_t> FeedNewData(const std::string& key, value_t value) {
  static std::unordered_map<std::string, std::unique_ptr<Worker>> workers;

  if (!workers.count(key)) {
    workers[key] = std::make_unique<Worker>(key);
  }
  const auto& worker = workers.at(key);

  auto routine = [&worker](value_t value) {
    worker->Insert(value);
    return worker->prediction();
  };

  // The training procedure must not be interrupted. All values should be
  // handled in chronological order.
  worker->Lock();
  // Asynchronous worker returns a promise
  auto prediction = std::async(std::launch::async, routine, value);
  worker->Unlock();

  return prediction;
}

// On caller side, do something with the returned promise like this
void Consume(const std::string& key, std::future<value_t>&& prediction) {
  auto routine = [](const std::string& key, std::future<value_t>&& prediction) {
    static std::ofstream output_file(OUTPUT_PATH);
    output_file << key << " " << prediction.get() << std::endl;
  };

  auto done = std::async(
      std::launch::async, routine, key, std::move(prediction));
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

#if VERBOSE
  std::cout << "Server started.\n";
  auto start_time = high_resolution_clock::now();
#endif

  std::ifstream input_file(INPUT_PATH);
  while (!input_file.eof()) {
    std::string key;
    value_t value;
    input_file >> key >> value;
    if (key == END_MARK) break;

    auto prediction = FeedNewData(key, value);

    // Consume(key, std::move(prediction));

    // Sleep for a while before reading next value
    // std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_TIME));
  }
  input_file.close();

#if VERBOSE
  auto end_time = high_resolution_clock::now();
  auto time = duration<double, std::milli>(end_time - start_time).count();
  std::cout << "Time: " << time << " ms\n";
#endif
}
