
import torch
import triton
import triton.language as tl


@triton.jit
def quant_symmetric_per_tensor_fp8e4nv(x_ptr, scale_ptr, out_ptr, LENGTH: tl.constexpr):
    vec_len = tl.arange(0, LENGTH)
    x = tl.load(x_ptr + vec_len).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    x_scaled = x / scale[:]
    x_clipped = tl.clamp(x_scaled, -448.0, 448.0) #.to(tl.bfloat16)
    # x_clipped = x_scaled.to(tl.bfloat16)
    tl.store(out_ptr + vec_len, x_clipped.to(tl.float8e4nv))
    # tl.store(out_ptr + vec_len, x_clipped)


# data_shape = (1024, )
data_shape = (8, )
# data_shape = (1, )
# data_shape = (2, )

torch.manual_seed(4711)
input_data = torch.randn(data_shape, device='cuda:0', dtype=torch.bfloat16)
out_data = torch.zeros(data_shape, dtype=torch.float8_e4m3fn, device='cuda:0')
# out_data = torch.zeros(data_shape, dtype=torch.bfloat16, device='cuda:0')

print(input_data)
scale = (input_data.amax() / 64.0).to(torch.float32)
# scale = torch.full(data_shape, (input_data.amax() / 64.0), device='cuda:0', dtype=torch.float32)
print(scale)

quant_symmetric_per_tensor_fp8e4nv[(1,)](input_data, scale, out_data, LENGTH=input_data.numel())

ref_data = (input_data / scale).to(torch.float8_e4m3fn)
# ref_data = (input_data / scale).to(torch.bfloat16)

print(ref_data)
print(out_data)

torch.testing.assert_close(ref_data, out_data)
