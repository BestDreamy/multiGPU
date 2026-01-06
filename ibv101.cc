#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <infiniband/verbs.h>

#define PAGE_SIZE 4096
struct pingpong_context {
	struct ibv_context	*context;
	struct ibv_comp_channel *channel;
	struct ibv_pd		*pd;
	struct ibv_mr		*mr;
	struct ibv_dm		*dm;
	struct ibv_cq		*cq;
	struct ibv_qp		*qp;
	char		*buf;
	int			 size;
	int			 send_flags;
	int			 rx_depth;
	int			 pending;
	struct ibv_port_attr     portinfo;
	// uint64_t		 completion_timestamp_mask;
};

struct ibv_device *get_the_first_ibv_device() {
	int num_devices = 0;
	struct ibv_device **device_list = ibv_get_device_list(&num_devices);

	printf("Total IB devices found: %d\n", num_devices);
	for (int i = 0; i < num_devices; i++) {
		printf("Device %d: %s\n", i, ibv_get_device_name(device_list[i]));
	}

	struct ibv_device *first_device = device_list[0];
	ibv_free_device_list(device_list);
	return first_device;
}

// Initialize rdma resources
static struct pingpong_context *init_pp_ctx(struct ibv_device *ib_dev) {
	struct pingpong_context *pp_ctx = (struct pingpong_context *)malloc(sizeof(struct pingpong_context));

	// 1. Open device context
	pp_ctx->context = ibv_open_device(ib_dev);

	pp_ctx->size       = PAGE_SIZE;
	pp_ctx->send_flags = IBV_SEND_SIGNALED;
	pp_ctx->rx_depth   = 1;
	pp_ctx->buf = (char *)aligned_alloc(PAGE_SIZE, pp_ctx->size);
	memset(pp_ctx->buf, 0, pp_ctx->size);

	// 2. Allocate Protection Domain
	pp_ctx->pd = ibv_alloc_pd(pp_ctx->context);

	// 3. Register Memory Region
	pp_ctx->mr = ibv_reg_mr(
		pp_ctx->pd, 
		pp_ctx->buf, 
		pp_ctx->size,
		IBV_ACCESS_LOCAL_WRITE // use in send/recv
		/*
			IBV_ACCESS_LOCAL_WRITE = 1 ,
			IBV_ACCESS_REMOTE_WRITE = (1<<1),
			IBV_ACCESS_REMOTE_READ = (1<<2),
			IBV_ACCESS_REMOTE_ATOMIC = (1<<3),
			IBV_ACCESS_MW_BIND = (1<<4)
		*/
	);

	int num_comp_vectors = pp_ctx->context->num_comp_vectors;

	// 4. Create Completion Queue
	pp_ctx->cq = ibv_create_cq(
		pp_ctx->context, 
		pp_ctx->rx_depth, 
		NULL, 
		NULL, 
		num_comp_vectors - 1
	);

	// 5. Create Queue Pair
	struct ibv_qp_init_attr qp_init_attr = {
		.send_cq = pp_ctx->cq,
		.recv_cq = pp_ctx->cq,
		.cap = {
			.max_send_wr = pp_ctx->rx_depth,
			.max_recv_wr = pp_ctx->rx_depth,
			.max_send_sge = 1,
			.max_recv_sge = 1,
		},
		// .max_inline_data = 0,
		.qp_type = IBV_QPT_RC
	};
	pp_ctx->qp = ibv_create_qp(pp_ctx->pd, &qp_init_attr);
	
	// 6. Change QP state
	struct ibv_device_attr dev_attr;
	ibv_query_device(pp_ctx->context, &dev_attr);

    struct ibv_port_attr port_attr;
	int port = 1;
    for (port = 1; port <= dev_attr.phys_port_cnt; ++port) {
        if (ibv_query_port(pp_ctx->context, port, &port_attr))
            continue;
        if (port_attr.state == IBV_PORT_ACTIVE) {
            printf("Using active port %d\n", port);
            break;
        }
    }

	struct ibv_qp_attr qp_attr = {
		.qp_state        = IBV_QPS_INIT,
		.qp_access_flags = 0,
		.pkey_index      = 0,
		.port_num        = port,
	};

	ibv_modify_qp(pp_ctx->qp, &qp_attr, 
		IBV_QP_STATE | 
		IBV_QP_PKEY_INDEX | 
		IBV_QP_PORT | 
		IBV_QP_ACCESS_FLAGS
	);

	return pp_ctx;
}

static int server_post_recv(struct pingpong_context *ctx)
{
	struct ibv_sge list = {
		.addr	= (uintptr_t)ctx->buf,
		.length = ctx->size,
		.lkey	= ctx->mr->lkey
	};
	struct ibv_recv_wr wr = {
		.wr_id	    = 0, //
		.sg_list    = &list,
		.num_sge    = 1,
	};

	struct ibv_recv_wr *bad_wr;

	ibv_post_recv(ctx->qp, &wr, &bad_wr);

	return 0;
}

int main() {
	struct ibv_device *ibv_device = get_the_first_ibv_device();

	struct pingpong_context *pp_ctx = init_pp_ctx(ibv_device);

	server_post_recv(pp_ctx);
}