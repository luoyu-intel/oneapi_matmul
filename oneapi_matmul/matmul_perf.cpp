/*******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include <oneapi/mkl.hpp>

using namespace dnnl;
using namespace sycl;

using tag = memory::format_tag;
using dt = memory::data_type;

struct gemm_dims_t {
	memory::dim m, n, k;
};

static const int min_runs = 4;

const char* get_type_string(dt type) {
	const char* type_string = "unknown";

#define TYPE_CASE(T) \
    if (type == dt::T) type_string = #T;
	TYPE_CASE(f16);
	TYPE_CASE(f32);
	TYPE_CASE(f64);
	TYPE_CASE(bf16);
	TYPE_CASE(s8);
	TYPE_CASE(u8);
#undef TYPE_CASE

	return type_string;
}

void print_test_case(dt type, gemm_dims_t dims) {
	std::cout << '[' << std::setw(4) << get_type_string(type);
	if (dims.m == dims.n && dims.m == dims.k)
		std::cout << " m = n = k = " << dims.m;
	else
		std::cout << " m = " << dims.m << ", n = " << dims.n
		<< ", k = " << dims.k;
	std::cout << "] " << std::flush;
}

void fill_random(std::vector<float>& out, bool is_integer) {
	static std::vector<float> random_data_i, random_data_f;
	constexpr size_t nrand = 1037;

	if (random_data_i.empty() || random_data_f.empty()) {
		std::mt19937 generator;
		std::uniform_int_distribution<int> dist_i(-16, 15);
		std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

		random_data_i.resize(nrand);
		for (auto& d : random_data_i)
			d = static_cast<float>(dist_i(generator));

		random_data_f.resize(nrand);
		for (auto& d : random_data_f)
			d = dist_f(generator);
	}

	auto& rd = is_integer ? random_data_i : random_data_f;

	for (size_t i = 0; i < out.size(); i += nrand) {
		size_t chunk = std::min(nrand, out.size() - i);
		std::memcpy(&out[i], rd.data(), chunk * sizeof(float));
	}
}


class DnnlGemmWrapper {
public:
	template<typename T>
	static constexpr dt todt() {
		if constexpr (std::is_same_v<T, float>) return dt::f32;
		else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
		else if constexpr (std::is_same_v<T, oneapi::mkl::bfloat16>) return dt::bf16;
		else static_assert(0);
	}

	static inline void row_gemm(const dnnl::engine& eng, dnnl::stream& stream, oneapi::mkl::transpose a_trans,
		oneapi::mkl::transpose b_trans, int m, int n, int k,
		const void* a, int lda, dt at, const void* b, int ldb, dt bt, void* c, int ldc, dt ct)
	{
		memory::dims a_dims = { m, k };
		memory::dims b_dims = { k, n };
		memory::dims c_dims = { m, n };
		using trans = oneapi::mkl::transpose;
		const auto a_in_md = memory::desc(a_dims, at, a_trans == trans::N ? tag::ab : tag::ba);
		const auto b_in_md = memory::desc(a_dims, bt, b_trans == trans::N ? tag::ab : tag::ba);
		const auto c_md = memory::desc(c_dims, ct, tag::ab);
		auto a_mem = memory(a_in_md, eng, (void*)a);
		auto b_mem = memory(b_in_md, eng, (void*)b);
		auto matmul_pd = matmul::primitive_desc(eng, a_in_md, b_in_md, c_md);
		auto c_mem = memory(matmul_pd.dst_desc(), eng, c);

		// Create the primitive.
		auto matmul_prim = matmul(matmul_pd);
		// Primitive arguments.
		std::unordered_map<int, memory> matmul_args;
		matmul_args.insert({ DNNL_ARG_SRC, a_mem });
		matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });
		matmul_args.insert({ DNNL_ARG_DST, c_mem });

		// Warmup executions.
		matmul_prim.execute(stream, matmul_args);
	}
};

double run_case(engine::kind engine_kind, dt type, gemm_dims_t dims,
	double time_limit = 0.) {
	bool is_integer = (type == dt::s8 || type == dt::u8);
	bool quick_test = (time_limit == 0.);

	// Create execution dnnl::engine.
	dnnl::engine engine(engine_kind, 0);

	// Create dnnl::stream.
	dnnl::stream engine_stream(engine);

	// Source (A), weights (B), and destination (C) matrix dimensions.
	memory::dims a_dims = { dims.m, dims.k };
	memory::dims b_dims = { dims.k, dims.n };
	memory::dims c_dims = { dims.m, dims.n };

	// Allocate buffers and random-initialize A/B
	std::vector<float> a_data(product(a_dims));
	std::vector<float> b_data(product(b_dims));
	std::vector<float> c_data(product(c_dims));

	fill_random(a_data, is_integer);
	fill_random(b_data, is_integer);

	// Create memory descriptors and memory objects for src, weights, bias, and
	// dst.
	auto c_md = memory::desc(c_dims, type, tag::ab);

	auto a_in_md = memory::desc(a_dims, type, tag::ab);
	auto b_in_md = memory::desc(b_dims, type, tag::ba);

	auto a_in_mem = memory(a_in_md, engine);
	auto b_in_mem = memory(b_in_md, engine);

	// Write data to memory object's handles.
	write_to_dnnl_memory(a_data.data(), a_in_mem);
	write_to_dnnl_memory(b_data.data(), b_in_mem);

	// Create primitive descriptor.
	auto matmul_pd = matmul::primitive_desc(engine, a_in_md, b_in_md, c_md);

	auto c_mem = memory(matmul_pd.dst_desc(), engine);

	// Create the primitive.
	auto matmul_prim = matmul(matmul_pd);

	// Start output.
	if (!quick_test) print_test_case(type, dims);

	// Primitive arguments.
	std::unordered_map<int, memory> matmul_args;
	matmul_args.insert({ DNNL_ARG_SRC, a_in_mem });
	matmul_args.insert({ DNNL_ARG_WEIGHTS, b_in_mem });
	matmul_args.insert({ DNNL_ARG_DST, c_mem });

	// Warmup executions.
	matmul_prim.execute(engine_stream, matmul_args);
	engine_stream.wait();

	auto start_first = std::chrono::steady_clock::now();
	matmul_prim.execute(engine_stream, matmul_args);
	engine_stream.wait();
	auto end_first = std::chrono::steady_clock::now();

	std::chrono::duration<double> dur_first = end_first - start_first;

	if (quick_test) return dur_first.count();

	int runs = std::max(min_runs, int(time_limit / dur_first.count()));

	// Timing runs.
	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i <= runs; i++)
#if 0
		matmul_prim.execute(engine_stream, matmul_args);
#else
		DnnlGemmWrapper::row_gemm(engine, engine_stream, oneapi::mkl::transpose::N, oneapi::mkl::transpose::T
			, dims.m, dims.n, dims.k, a_in_mem.get_data_handle(), dims.k, type,
			b_in_mem.get_data_handle(), dims.k, type, c_mem.get_data_handle(), dims.n, type);
#endif
	engine_stream.wait();

	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> duration = end - start;

	// Display the result.
	double avg_time = (duration.count() - dur_first.count()) / runs;
	double total_ops = double(dims.m) * double(dims.n) * double(dims.k) * 2;
	double perf = (total_ops / avg_time) * 1e-9;

	auto scale_string = "G";
	auto unit_string = is_integer ? "Op/s" : "Flop/s";

	if (perf >= 1000) {
		perf /= 1000;
		scale_string = "T";
	}

	std::cout << perf << ' ' << scale_string << unit_string << std::endl;

	return avg_time;
}

template<typename T>
double run_mkl_case(gemm_dims_t dims, double time_limit = 0.) {
	bool quick_test = (time_limit == 0.);
	int m = dims.m;
	int n = dims.n;
	int k = dims.k;
	auto selector = sycl::default_selector_v;
	// Create an exception handler for asynchronous SYCL exceptions
	static auto exception_handler = [](sycl::exception_list e_list) {
		for (std::exception_ptr const& e : e_list) {
			try {
				std::rethrow_exception(e);
			}
			catch (std::exception const& e) {
#if _DEBUG
				std::cout << "Failure" << std::endl;
#endif
				std::terminate();
			}
		}
		};

	queue q(selector, exception_handler);

	// Source (A), weights (B), and destination (C) matrix dimensions.
	memory::dims a_dims = { dims.m, dims.k };
	memory::dims b_dims = { dims.k, dims.n };
	memory::dims c_dims = { dims.m, dims.n };

	// Allocate buffers and random-initialize A/B
	std::vector<float> a_data(product(a_dims));
	std::vector<float> b_data(product(b_dims));
	std::vector<float> c_data(product(c_dims));

	fill_random(a_data, false);
	fill_random(b_data, false);
	// Create buffers that hold the data shared between the host and the devices.
		// The buffer destructor is responsible to copy the data back to host when it
		// goes out of scope.
	auto a_buf = malloc_device<T>(m * k, q);
	auto b_buf = malloc_device<T>(n * k, q);
	auto c_buf = malloc_device<T>(m * n, q);

	q.memcpy(a_buf, a_data.data(), m * k * sizeof(T)).wait();
	q.memcpy(b_buf, b_data.data(), n * k * sizeof(T)).wait();
	q.memset(c_buf, 0, m * n * sizeof(T)).wait();
	using trans = oneapi::mkl::transpose;

	// Start output.
	if (!quick_test) print_test_case(dt::f32, dims);
	oneapi::mkl::blas::row_major::gemm(q, trans::N, trans::T, m, n, k, 1.f, a_buf, k, b_buf, k, 0.f
		, c_buf, n, oneapi::mkl::blas::compute_mode::prefer_alternate);
	q.wait();
	auto start_first = std::chrono::steady_clock::now();
	oneapi::mkl::blas::row_major::gemm(q, trans::N, trans::T, m, n, k, 1.f, a_buf, k, b_buf, k, 0.f
		, c_buf, n, oneapi::mkl::blas::compute_mode::prefer_alternate);
	q.wait();
	auto end_first = std::chrono::steady_clock::now();

	std::chrono::duration<double> dur_first = end_first - start_first;

	if (quick_test) return dur_first.count();

	int runs = std::max(min_runs, int(time_limit / dur_first.count()));

	// Timing runs.
	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i <= runs; i++)
		oneapi::mkl::blas::row_major::gemm(q, trans::N, trans::T, m, n, k, 1.f, a_buf, k, b_buf, k, 0.f
			, c_buf, n, oneapi::mkl::blas::compute_mode::prefer_alternate);
	q.wait();

	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> duration = end - start;

	// Display the result.
	double avg_time = (duration.count() - dur_first.count()) / runs;
	double total_ops = double(dims.m) * double(dims.n) * double(dims.k) * 2;
	double perf = (total_ops / avg_time) * 1e-9;

	auto scale_string = "G";
	auto unit_string = "Flop/s";

	if (perf >= 1000) {
		perf /= 1000;
		scale_string = "T";
	}

	std::cout << "MKL: " << perf << ' ' << scale_string << unit_string << std::endl;
	sycl::free(a_buf, q);
	sycl::free(b_buf, q);
	sycl::free(c_buf, q);
	return avg_time;
}

void run(engine::kind engine_kind, dt type, gemm_dims_t dims,
	double time_limit) {
	try {
		if (dims.m * dims.n != 0) {
			// Dimensions manually specified by user.
			run_case(engine_kind, type, dims, time_limit);
		}
		else {
			// Automatically choose dimensions to fit time limit.
			int mnk = 128;
			const int max_mnk = 8192;

			while (mnk < max_mnk) {
				dims.m = dims.n = dims.k = mnk;
				double time1 = run_case(engine_kind, type, dims);
				double nruns_est = std::max(1., time_limit / time1);
				double mnk_expand = std::exp2(
					std::round(std::log2(nruns_est / min_runs) / 3.));
				if (mnk_expand <= 1) break;
				mnk = static_cast<int>(
					std::min<double>(max_mnk, mnk * mnk_expand));
			}

			dims.m = dims.n = dims.k = mnk;
			run_case(engine_kind, type, dims, time_limit);
		}
		if (type == dt::f32)
		{
			run_mkl_case<float>(dims, time_limit);
		}
		else if (type == dt::f16) {
			run_mkl_case<sycl::half>(dims, time_limit);
		}
		else if (type == dt::bf16) {
			run_mkl_case<oneapi::mkl::bfloat16>(dims, time_limit);
		}
	}
	catch (dnnl::error& e) {
		// Catch and report unimplemented cases.
		if (e.status == dnnl_unimplemented) {
			print_test_case(type, dims);
			std::cout << "unsupported" << std::endl;
		}
		else
			throw;
	}
}

void bad_args() {
	std::cerr << "Usage: matmul-perf-cpp [cpu|gpu]\n"
		"       matmul-perf-cpp [cpu|gpu] <size>\n"
		"       matmul-perf-cpp [cpu|gpu] <m> <n> <k>\n"
		"If a single <size> is specified, it is used for all three "
		"dimensions (m/n/k).\n";
	throw std::invalid_argument("Incorrect input arguments.");
}

void matmul_perf(engine::kind engine_kind, int argc, char** argv) {
	gemm_dims_t dims = { 0, 0, 0 };

	if (argc > 2) {
		if (argc == 3)
			dims.m = dims.n = dims.k = std::atoi(argv[2]);
		else if (argc == 5) {
			dims.m = std::atoi(argv[2]);
			dims.n = std::atoi(argv[3]);
			dims.k = std::atoi(argv[4]);
		}
		else
			bad_args();

		if (dims.m <= 0 || dims.n <= 0 || dims.k <= 0) bad_args();
	}

	run(engine_kind, dt::f32, dims, 2.0);
	run(engine_kind, dt::f16, dims, 2.0);
	run(engine_kind, dt::bf16, dims, 2.0);
	run(engine_kind, dt::s8, dims, 2.0);
}

void run_check(engine::kind engine_kind, dt type, gemm_dims_t dims) {
	bool is_integer = (type == dt::s8 || type == dt::u8);

	// Create execution dnnl::engine.
	dnnl::engine engine(engine_kind, 0);

	// Create dnnl::stream.
	dnnl::stream engine_stream(engine);

	// Source (A), weights (B), and destination (C) matrix dimensions.
	memory::dims a_dims = { dims.m, dims.k };
	memory::dims b_dims = { dims.k, dims.n };
	memory::dims c_dims = { dims.m, dims.n };

	// Allocate buffers and random-initialize A/B
	std::vector<float> a_data(product(a_dims));
	std::vector<float> b_data(product(b_dims));
	std::vector<float> c_data(product(c_dims));
	std::vector<float> c_data_mkl(product(c_dims));

	fill_random(a_data, is_integer);
	fill_random(b_data, is_integer);

	// Create memory descriptors and memory objects for src, weights, bias, and
	// dst.
	auto c_md = memory::desc(c_dims, type, tag::ab);

	auto a_in_md = memory::desc(a_dims, dt::f32, tag::ab);
	auto b_in_md = memory::desc(b_dims, dt::f32, tag::ba);

	auto a_in_mem = memory(a_in_md, engine);
	auto b_in_mem = memory(b_in_md, engine);

	// Write data to memory object's handles.
	write_to_dnnl_memory(a_data.data(), a_in_mem);
	write_to_dnnl_memory(b_data.data(), b_in_mem);

	// Create primitive descriptor.
	auto matmul_pd = matmul::primitive_desc(engine, a_in_md, b_in_md, c_md);

	auto c_mem = memory(matmul_pd.dst_desc(), engine);

	// Create the primitive.
	auto matmul_prim = matmul(matmul_pd);


	// Primitive arguments.
	std::unordered_map<int, memory> matmul_args;
	matmul_args.insert({ DNNL_ARG_SRC, a_in_mem });
	matmul_args.insert({ DNNL_ARG_WEIGHTS, b_in_mem });
	matmul_args.insert({ DNNL_ARG_DST, c_mem });

	// Warmup executions.
	matmul_prim.execute(engine_stream, matmul_args);
	engine_stream.wait();
	read_from_dnnl_memory(c_data.data(), c_mem);

	queue q = sycl_interop::get_queue(engine_stream);
	int m = dims.m;
	int n = dims.n;
	int k = dims.k;
	auto a_buf = malloc_device<float>(m * k, q);
	auto b_buf = malloc_device<float>(n * k, q);
	auto c_buf = malloc_device<float>(m * n, q);
	using trans = oneapi::mkl::transpose;

	q.memcpy(a_buf, a_data.data(), m * k * sizeof(float)).wait();
	q.memcpy(b_buf, b_data.data(), n * k * sizeof(float)).wait();
	q.memset(c_buf, 0, m * n * sizeof(float)).wait();
	oneapi::mkl::blas::row_major::gemm(q, trans::N, trans::T, m, n, k, 1.f, a_buf, k, b_buf, k, 0.f
		, c_buf, n, oneapi::mkl::blas::compute_mode::prefer_alternate);
	q.wait();
	q.memcpy(c_data_mkl.data(), c_buf, n * m * sizeof(float)).wait();
	double error = 0;
	for (size_t i = 0; i < n * m; i++)
	{
		error += abs(c_data_mkl[i] - c_data[i]);
	}
	std::cout << "Error: " << error << std::endl;
}

int main(int argc, char** argv) {
	run_check(engine::kind::gpu, dt::f32, { 1024,4096,4096 });
	return handle_example_errors(
		matmul_perf, parse_engine_kind(argc, argv, 3), argc, argv);
}
