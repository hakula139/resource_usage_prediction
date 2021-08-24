#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common/config.hpp"
#include "common/types.hpp"
#include "worker.hpp"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

std::unordered_map<std::string, std::unique_ptr<Worker>> workers;
std::mutex workers_lk;

std::atomic_int64_t n_threads = 0;
std::mutex n_threads_lk;
std::condition_variable n_threads_cv;

// Feed the latest data and get a prediction for the next value.
value_t Feed(const std::string& key, value_t value) {
  {
    // Prevent conflicts when creating a new worker.
    std::unique_lock<std::mutex> guard{workers_lk};

    if (!workers.count(key)) {
      workers[key] = std::make_unique<Worker>(key);
    }
  }

  const auto& worker = workers.at(key);

  // The training procedure must not be interrupted. All values should be
  // processed in chronological order.
  std::unique_lock<std::mutex> guard{worker->mutex()};

  worker->Insert(value);
  return worker->prediction();
}

// An example for caller side.
void Consume(const std::string& key, value_t value) {
  auto start_time = high_resolution_clock::now();

  auto routine = [start_time](const std::string& key, value_t value) {
    auto prediction = Feed(key, value);

    auto end_time = high_resolution_clock::now();
    auto time = duration<double, std::milli>(end_time - start_time).count();

    std::cout << key << " " << prediction;
    std::cout << " \tTime: " << std::fixed << std::setprecision(4)
              << std::setw(7) << time << " ms";

    {
      std::unique_lock<std::mutex> guard(n_threads_lk);
      std::cout << " | Threads: " << n_threads << std::endl;
      --n_threads;
    }
    // Unblock the producer thread.
    n_threads_cv.notify_one();
  };

  std::thread consumer{routine, key, value};
  consumer.detach();
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  std::cout << "Server started.\n";

  std::ifstream input_file(INPUT_PATH);

  while (!input_file.eof()) {
    std::string key;
    value_t value;
    input_file >> key >> value;
    if (key == END_MARK) break;

    {
      // Limit the number of running threads.
      std::unique_lock<std::mutex> guard(n_threads_lk);
      n_threads_cv.wait(guard, [] { return n_threads < MAX_N_THREADS; });
      ++n_threads;
    }

    Consume(key, value);

    // Sleep for a while before reading next value.
    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_TIME));
  }

  input_file.close();
}
