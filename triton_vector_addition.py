import torch 
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  block_start = pid*BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets<n_elements
  x = tl.load(x_ptr + offsets, mask = mask)
  y = tl.load(y_ptr + offsets, mask = mask)
  output = x+y
  tl.store(output_ptr + offsets, output, mask=mask)

def add(x : torch.Tensor, y : torch.Tensor):
  output = torch.empty_like(x)
  assert x.device == DEVICE and y.device==DEVICE and output.device==DEVICE
  n_elements = output.numel()
  grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
  add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
  return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device = DEVICE)
output_torch = x + y
output_triton = add(x,y)
print(output_torch)
print(output_triton)
print(torch.max(torch.abs(output_torch- output_triton)))


# output
# tensor([1.3713, 1.3076, 0.4940,  ..., 0.9360, 1.3598, 0.3215], device='cuda:0')
# tensor([1.3713, 1.3076, 0.4940,  ..., 0.9360, 1.3598, 0.3215], device='cuda:0')
# tensor(0., device='cuda:0')

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],  
        x_log=True,  
        line_arg='provider',  
        line_vals=['triton', 'torch'],  
        line_names=['Triton', 'Torch'],  
        styles=[('blue', '-'), ('green', '-')],  
        ylabel='GB/s',  
        plot_name='vector-add-performance',
        args={}, 
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)


# vector-add-performance:
#            size      Triton       Torch
# 0        4096.0    8.000000    8.000000
# 1        8192.0   15.999999   15.999999
# 2       16384.0   27.428571   27.428571
# 3       32768.0   54.857142   54.613333
# 4       65536.0   96.000000   85.333330
# 5      131072.0  139.636363  139.636363
# 6      262144.0  153.600004  153.600004
# 7      524288.0  204.800005  204.800005
# 8     1048576.0  180.705879  180.705879
# 9     2097152.0  211.862064  210.051276
# 10    4194304.0  216.528637  215.578943
# 11    8388608.0  221.905193  222.911559
# 12   16777216.0  227.292483  226.768160
# 13   33554432.0  229.682243  229.548160
# 14   67108864.0  231.337547  230.963871
# 15  134217728.0  231.866143  232.088538
