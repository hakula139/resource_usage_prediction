#include <fstream>
#include <iostream>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <unordered_map>

#include "common/config.hpp"
#include "common/types.hpp"
#include "worker.hpp"

int main() {
  std::cout << "Server started.\n";

  std::ifstream input_file(INPUT_PATH);
  std::unordered_map<std::string, std::unique_ptr<Worker>> workers;

  while (!input_file.eof()) {
    std::string key;
    value_t value;
    input_file >> key >> value;
    if (key == END_MARK) break;

    if (!workers.count(key)) {
      workers[key] = std::make_unique<Worker>(key);
    }

    const auto& worker = workers.at(key);
    worker->Insert(value);

    // Sleep for a while before reading next value
    // std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_TIME));
  }

  input_file.close();
}
