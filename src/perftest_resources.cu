#include <stdint.h>


template <typename T>
__global__
void bufcmp(size_t buf_size, int *result, T *buf, T *cmp, size_t cmp_len, int is_lat)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < buf_size / sizeof(*buf)) {
		if (buf[i] != cmp[i % (cmp_len / sizeof(*buf))]) {
			*result = 1;
		}
		/* poison 8 bytes of the buffers every 64K. (GPU BAR1 mapping granularity) */
		if (i % (65536 / sizeof(*buf)) < 8 / sizeof(*buf)) {
			if (is_lat == 1) {
				/* In the latency case, clear the buffer that was written into */
				buf[i] = 0;
			} else {
				/* In the BW case, clear the verification buffer
				 * and rotate the bytes in the write buffer to
				 * avoid masking failures between iterations.
				 */
				cmp[i] = 0;
				if (i < (buf_size / sizeof(*buf)) - 1) {
					buf[i] = buf[(i + 1) % buf_size];
				}
			}
		}
	}
}

/*
 * Launch the kernel for comparing a GPU buffer vs. another expected/derived value.
 *
 * args:
 * buf - In the BW case, where we do client-side validation, this is the client-side buffer we write from. In the LAT case
 * this is the part of the buffer which gets written into. IN both cases, this value corresponds to part, or all of the buf
 * value in the workload's pingpong_context struct.
 * buf_size - the length of buf to compare.
 * cmp - In the BW case, where we do client-side validation, this is the client side buffer we read into from the remote
 * server buffer we just wrote from. It is the same size as buf. In the LAT case, it is a repeating byte pattern defined
 * at startup and of a much shorter length than the buffer.
 * cmp_len - The length of cmp to compare. This is assumed to be <= buf_size.
 * is_lat - flag to indicate if we are in a LAT or BW workload. Affects the way we poison the buffers.
 *
 */
extern "C" void start_bufcmp_gpu_async(char *buf, size_t buf_size, char *cmp, size_t cmp_len, int *result, int is_lat)
{
	size_t num_iterations;

	/* If both buffers don't have a length that is an even multiple of 8 bytes, fall back to a bytewise comparison. */
	if (buf_size < 8 || buf_size % sizeof(uint64_t) != 0 || cmp_len % sizeof(uint64_t) != 0) {
			bufcmp<<<(buf_size + 1023) / 1024, 1024>>>(buf_size, result, buf, cmp, cmp_len, is_lat);
	} else {
		num_iterations = buf_size / sizeof(uint64_t);
		bufcmp<<<(num_iterations + 1023) / 1024, 1024>>>(buf_size, result, (uint64_t *)buf, (uint64_t *)cmp, cmp_len, is_lat);
	}

}
