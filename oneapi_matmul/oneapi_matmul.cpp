// oneapi_matmul.cpp : Defines the entry point for the application.
//

#include "oneapi_matmul.h"
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

using namespace std;
using namespace sycl;

template<typename T>
void mklgemm(queue& q, const std::vector<T>& ah, const std::vector<T>& bh,
	std::vector<T>& ch, int m, int n, int k) {

	// Create buffers that hold the data shared between the host and the devices.
	// The buffer destructor is responsible to copy the data back to host when it
	// goes out of scope.
	auto a_buf = malloc_device<T>(m * k, q);
	auto b_buf = malloc_device<T>(n * k, q);
	auto c_buf = malloc_device<T>(m * n, q);

	q.memcpy(a_buf, ah.data(), m * k * sizeof(T)).wait();
	q.memcpy(b_buf, bh.data(), n * k * sizeof(T)).wait();
	q.memset(c_buf, 0, m * n * sizeof(T)).wait();

	timer<milliseconds> tm;
	tm.start();
	using trans = oneapi::mkl::transpose;
	oneapi::mkl::blas::row_major::gemm(q, trans::N, trans::N, m, n, k, 1.f, a_buf, k, b_buf, n, 0.f
		, c_buf, n, oneapi::mkl::blas::compute_mode::unset);
	// Wait until compute tasks on GPU done
	q.wait();
	auto tms = tm.stop();
	int iters = 1000.f / tms;
	iters = std::max(1, iters);
	iters = std::min(1000, iters);
	q.memcpy(ch.data(), c_buf, m * n * sizeof(T)).wait();

	tm.start();
	for (size_t i = 0; i < iters; i++)
	{
		oneapi::mkl::blas::row_major::gemm(q, trans::N, trans::N, m, n, k, 1.f, a_buf, k, b_buf, n, 0.f
			, c_buf, n, oneapi::mkl::blas::compute_mode::prefer_alternate);
	}
	q.wait();
	auto avgt = tm.stop() / iters;
	double FOPS = double(m) * k * n * 2;
	double flops = FOPS / avgt / 1e3;
	printf("%s %d (%d,%d,%d) time:%.2f ms FLOPs:%f\n", __FUNCTION__, __LINE__, m, n, k, avgt, flops);
	sycl::free(a_buf, q);
	sycl::free(b_buf, q);
	sycl::free(c_buf, q);
}

void sgemmtest(queue& q, int m, int n, int k) {
	std::vector<float> A(m * k), B(n * k), C(m * n), CCpu(m * n);
	mklgemm<float>(q, A, B, C, m, n, k);
}


void hgemmtest(queue& q, int m, int n, int k) {
	std::vector<half> A(m * k), B(n * k), C(m * n), CRef(m * n);
	mklgemm<half>(q, A, B, C, m, n, k);
}

int main1(int argc, char* argv[]) {
	// The default device selector will select the most performant device.
	auto selector = default_selector_v;
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

	// Print out the device information used for the kernel code.
	std::cout << "Running on device: "
		<< q.get_device().get_info<info::device::name>() << "\n";

	sgemmtest(q, 4096, 4096, 4096);
	hgemmtest(q, 4096, 4096, 4096);
	std::cout << "Vector add successfully completed on device.\n";
	return 0;
}