#include <stdio.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include "mlx5_ifc.h"
#include "perftest_parameters.h"
#include "perftest_resources.h"

#include NVML_PATH

#define WAIVED 2

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    (1UL << GPU_PAGE_SHIFT)

struct gpu_mem_handle {
	CUdeviceptr ptr; // aligned ptr if requested; otherwise, the same as unaligned_ptr.
	union {
		CUdeviceptr unaligned_ptr; // for tracking original ptr; may be unaligned.
#if CUDA_VERSION >= 11000
		// VMM with GDR support is available from CUDA 11.0
		CUmemGenericAllocationHandle handle;
#endif
	};
	size_t size;
	size_t allocated_size;
};

static CUresult gpu_mem_alloc(struct gpu_mem_handle *handle, 
	const size_t size, 
	bool aligned_mapping, 
	bool set_sync_memops)
{
	CUresult ret = CUDA_SUCCESS;
	CUdeviceptr ptr, out_ptr;
	size_t allocated_size;

	if (aligned_mapping)
		allocated_size = size + GPU_PAGE_SIZE - 1;
	else
		allocated_size = size;

	ret = cuMemAlloc(&ptr, allocated_size);
	if (ret != CUDA_SUCCESS)
		return ret;

	if (set_sync_memops) {
		unsigned int flag = 1;
		ret = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr);
		if (ret != CUDA_SUCCESS) {
			cuMemFree(ptr);
			return ret;
		}
	}

	if (aligned_mapping)
		out_ptr = ROUND_UP(ptr, GPU_PAGE_SIZE);
	else
		out_ptr = ptr;

	handle->ptr = out_ptr;
	handle->unaligned_ptr = ptr;
	handle->size = size;
	handle->allocated_size = allocated_size;

	return CUDA_SUCCESS;
}

static CUresult gpu_mem_free(struct gpu_mem_handle *handle)
{
	CUresult ret = CUDA_SUCCESS;

	ret = cuMemFree(handle->unaligned_ptr);
	if (ret == CUDA_SUCCESS)
		memset(handle, 0, sizeof(*handle));

	return ret;
}

#define NVIDIA_DRIVER_VERSION_MAX_LENGTH 80
static int get_nvidia_driver_version(int *major_version)
{
	static int cached_version = -1;
	char version_string[NVIDIA_DRIVER_VERSION_MAX_LENGTH];
	char c;
	int i;

	if (cached_version < 0) {
		if (nvmlInit() != NVML_SUCCESS) {
			fprintf(stderr, " Failed to initialize NVML\n");
			return FAILURE;
		}

		if (nvmlSystemGetDriverVersion(version_string, NVIDIA_DRIVER_VERSION_MAX_LENGTH) != NVML_SUCCESS) {
			fprintf(stderr, " Failed to get the nvidia driver version string\n");
			return FAILURE;
		}

		for (i = 0; i < NVIDIA_DRIVER_VERSION_MAX_LENGTH; ++i) {
			c = version_string[i];
			if (c < '0' || c > '9') {
				version_string[i] = '\0';
				break;
			}
		}

		cached_version = atoi(version_string);
	}

	*major_version = cached_version;

	return 0;
}

#define NVIDIA_DRIVER_PARAMS_FILE_PATH "/proc/driver/nvidia/params"
#define NVIDIA_DRIVER_PARAMS_LINE_INITIAL_LENGTH 100
static int get_peer_mapping_override_status(int *status)
{
	int ret = 0;
	int cached_status = -1;
	FILE *f = NULL;
	char *line = NULL;
	size_t line_length;
	int cont;

	if (cached_status < 0) {
		f = fopen(NVIDIA_DRIVER_PARAMS_FILE_PATH, "r");
		if (!f) {
			ret = FAILURE;
			fprintf(stderr, " Failed to open %s\n", NVIDIA_DRIVER_PARAMS_FILE_PATH);
			goto out;
		}

		line_length = NVIDIA_DRIVER_PARAMS_LINE_INITIAL_LENGTH;
		line = malloc(sizeof(char) * line_length);
		if (!line) {
			ret = FAILURE;
			fprintf(stderr, " Failed to malloc memory\n");
			goto out;
		}

		do {
			line_length = (line_length > NVIDIA_DRIVER_PARAMS_LINE_INITIAL_LENGTH) ? line_length : NVIDIA_DRIVER_PARAMS_LINE_INITIAL_LENGTH;
			cont = (getline(&line, &line_length, f) >= 0);
			if (strstr(line, "PeerMappingOverride=1")) {
				cached_status = 1;
				break;
			}
		} while (cont);

		if (cached_status < 0)
			cached_status = 0;
	}

	*status = cached_status;

out:
	if (line)
		free(line);

	if (f)
		fclose(f);

	return ret;
}

static const char *status_to_string(int status)
{
	switch (status) {
	case 0:
		return "PASSED";
	case 1:
		return "FAILED";
	case 2:
		return "WAIVED";
	default:
		return "UNKNOWN";
	}
}

static int control_in_vidmem_test(struct ibv_context *ib_ctx)
{
	int status = 0;
	struct gpu_mem_handle mhandle;
	struct mlx5dv_devx_umem *umem = NULL;
	int gmem_allocated = 0;
	int nvidia_driver_version;

	if (get_nvidia_driver_version(&nvidia_driver_version)) {
		status = FAILURE;
		goto out;
	}

	if (nvidia_driver_version < 510) {
		status = WAIVED;
		goto out;
	}   

	if (gpu_mem_alloc(&mhandle, GPU_PAGE_SIZE, GPU_PAGE_SIZE, true) != CUDA_SUCCESS) {
		status = FAILURE;
		goto out;
	}
	gmem_allocated = 1;

	umem = mlx5dv_devx_umem_reg(ib_ctx, (void *)mhandle.ptr, mhandle.size, IBV_ACCESS_LOCAL_WRITE);
	if (!umem) {
		status = FAILURE;
		goto out;
	}

out:
	if (umem)
		mlx5dv_devx_umem_dereg(umem);

	if (gmem_allocated)
		gpu_mem_free(&mhandle);

	return status;
}

static int gpu_uar_mapping_test(struct ibv_context *ib_ctx)
{
	int status = 0;
	struct mlx5dv_devx_uar *uar = NULL;
	uint8_t log_reg_size;
	size_t reg_size;
	uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {0,};
	uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {0,};
	int did_host_register = 0;
	int peer_mapping_override = 0;

	status = get_peer_mapping_override_status(&peer_mapping_override);
	if (status) {
		status = FAILURE;
		goto out;
	}

	if (!peer_mapping_override) {
		status = WAIVED;
		goto out;
	}

	uar = mlx5dv_devx_alloc_uar(ib_ctx, MLX5DV_UAR_ALLOC_TYPE_BF);
	if (!uar) {
		fprintf(stderr, " Failed in mlx5dv_devx_alloc_uar\n");
		status = FAILURE;
		goto out;
	}

	DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
	DEVX_SET(query_hca_cap_in, cmd_cap_in, op_mod,
		MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE |
		HCA_CAP_OPMOD_GET_CUR
	);

	status = mlx5dv_devx_general_cmd(ib_ctx, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out, sizeof(cmd_cap_out));
	if (status) {
		fprintf(stderr, " Failed in mlx5dv_devx_general_cmd\n");
		status = FAILURE;
		goto out;
	}

	log_reg_size = DEVX_GET(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap.log_bf_reg_size);

	// The size of 1st + 2nd half (as when we use alternating DB)
	reg_size = 1LLU << log_reg_size;

	if (cuMemHostRegister(uar->reg_addr, reg_size, CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_IOMEMORY) != CUDA_SUCCESS) {
		fprintf(stderr, " Failed in cuMemHostRegister\n");
		status = FAILURE;
		goto out;
	}
	did_host_register = 1;

out:
	if (did_host_register)
		cuMemHostUnregister(uar->reg_addr);

	if (uar)
		mlx5dv_devx_free_uar(uar);

	return status;
}

int main(int argc, char *argv[])
{
	int ret_parser;
	struct ibv_device *ib_dev = NULL;
	struct pingpong_context	ctx;
	struct perftest_parameters user_param;
	int control_in_vidmem_result = 0;
	int gpu_uar_mapping_result = 0;

	/* init default values to user's parameters */
	memset(&user_param, 0, sizeof(user_param));
	memset(&ctx, 0, sizeof(ctx));

	user_param.verb    = WRITE;
	user_param.tst     = BW;
	strncpy(user_param.version, VERSION, sizeof(user_param.version));

	/* Configure the parameters values according to user arguments or default values. */
	ret_parser = parser(&user_param, argv, argc);
	if (ret_parser) {
		if (ret_parser != VERSION_EXIT && ret_parser != HELP_EXIT)
			fprintf(stderr," Parser function exited with Error\n");
		return FAILURE;
	}

	if (!user_param.use_cuda) {
		fprintf(stderr, " This test requires CUDA\n");
		return FAILURE;
	}

	if (user_param.cuda_device_bus_id) {
		int err;

		printf("initializing CUDA\n");
		CUresult error = cuInit(0);
		if (error != CUDA_SUCCESS) {
			printf("cuInit(0) returned %d\n", error);
			return FAILURE;
		}

		printf("Finding PCIe BUS %s\n", user_param.cuda_device_bus_id);
		err = cuDeviceGetByPCIBusId(&user_param.cuda_device_id, user_param.cuda_device_bus_id);
		if (err != 0) {
			fprintf(stderr, "We have an error from cuDeviceGetByPCIBusId: %d\n", err);
			return FAILURE;
		}
		printf("Picking GPU number %d\n", user_param.cuda_device_id);
	}
	if (pp_init_gpu(&ctx, user_param.cuda_device_id)) {
		fprintf(stderr, "Couldn't init GPU context\n");
		return FAILURE;
	}

	ib_dev = ctx_find_dev(&user_param.ib_devname);
	if (!ib_dev) {
		fprintf(stderr," Unable to find the Infiniband/RoCE device\n");
		return FAILURE;
	}

	/* Getting the relevant context from the device */
	ctx.context = ibv_open_device(ib_dev);
	if (!ctx.context) {
		fprintf(stderr, " Couldn't get context for the device\n");
		return FAILURE;
	}


	control_in_vidmem_result = control_in_vidmem_test(ctx.context);
	gpu_uar_mapping_result = gpu_uar_mapping_test(ctx.context);

	printf(RESULT_LINE);
	printf("Control in vidmem: %s\n", status_to_string(control_in_vidmem_result));
	printf("GPU UAR mapping: %s\n", status_to_string(gpu_uar_mapping_result));

	return (control_in_vidmem_result == FAILURE || gpu_uar_mapping_result == FAILURE) ? FAILURE : SUCCESS;
}
