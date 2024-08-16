// oneapi_matmul.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include <chrono>
// TODO: Reference additional headers your program requires here.

using milliseconds = std::chrono::milliseconds;
using nanoseconds = std::chrono::nanoseconds;
using microseconds = std::chrono::microseconds;
template <typename _DUR = std::chrono::milliseconds>
class timer {
 public:
  using sclock_t = std::chrono::steady_clock;
  using stime_point_t = std::chrono::time_point<sclock_t>;

  timer() { clear(); }

  void start() { startT = sclock_t::now(); }

  void clear() { startT = stime_point_t::min(); }

  bool null_state() { return startT == stime_point_t::min(); }

  float stop() { return static_cast<float>(std::chrono::duration_cast<_DUR>(sclock_t::now() - startT).count()); }

  stime_point_t startT;
};

template <typename T>
class minmax_statistics {
 public:
  minmax_statistics() { clear(); }

  void clear() {
    min_val = std::numeric_limits<T>::max();
    max_val = std::numeric_limits<T>::min();
    avg_val = 0;
    count = 0;
  }

  void add(T _val) {
    min_val = min_val > _val ? _val : min_val;
    max_val = max_val < _val ? _val : max_val;
    count += 1;
    avg_val = (avg_val * (count - 1) + _val) / count;
  }

  T min_val, max_val, avg_val;
  size_t count;
};

template <int _PRINT_CYCLE_MS = 100, typename _PRECISION = microseconds, typename _LOG_PRECISION = milliseconds>
class timer_statistics_logger {
 public:
  typedef timer<milliseconds> log_timer_t;
  timer_statistics_logger() {
    clear();
    log_ratio = static_cast<float>(std::chrono::duration_cast<_PRECISION>(_LOG_PRECISION(1)).count());
  }

  void clear() {
    statis.clear();
    logtm.clear();
  }

  void start() {
    if (logtm.null_state()) {
      logtm.start();
    }
    tm.start();
  }

  bool stop() {
    auto elapsed = tm.stop();
    statis.add(elapsed);
    if (logtm.stop() >= _PRINT_CYCLE_MS) {
      record();
      clear();
      logtm.start();
      return true;
    }
    return false;
  }

  bool add(float time) {
    statis.add(time);
    if (logtm.stop() >= _PRINT_CYCLE_MS) {
      record();
      clear();
      logtm.start();
      return true;
    }
    return false;
  }

  const char* get_log_str() {
    sprintf(str, "Min:%.4f, Max:%.4f, Average:%.4f", min_val, max_val, avg_val);
    return str;
  }
  float min_val, max_val, avg_val;

  void record() {
    if (statis.count) {
      min_val = statis.min_val / log_ratio;
      max_val = statis.max_val / log_ratio;
      avg_val = statis.avg_val / log_ratio;
    }
  }

 private:
  float log_ratio;
  char str[256];
  timer<_PRECISION> tm;
  minmax_statistics<float> statis;
  timer<milliseconds> logtm;
};
