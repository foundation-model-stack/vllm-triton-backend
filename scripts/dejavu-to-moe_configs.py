import sys
import os
import json

# __vllm_base_path__ = '/home/zrlngl/watsonx/vllm/vllm/model_executor/layers/fused_moe/configs/'
# __vllm_base_path__ = '/home/zrlngl/watsonx/vllm/ngl_configs/'
__vllm_base_path__ = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ngl_configs/")
)

moe_keys = [
    "N",
    "K",
    "E",
    # 'EM',
    "num_valid_tokens",
    "num_actual_tokens",
    "stride_am",
    "stride_ak",
    "stride_be",
    "stride_bk",
    "stride_bn",
    "stride_cm",
    "stride_cn",
    "MUL_ROUTED_WEIGHT",
    "top_k",
    "compute_type",
    "use_fp8_w8a8",
    "use_int8_w8a16",
]
__skip_config_args__ = ["enable_persistent", "maxnreg"]


def moe_key_to_param_dict(k):
    kl = k[1:-1].split(", ")
    ret = {}
    for i, label in enumerate(moe_keys):
        ret[label] = kl[i]
    return ret


def create_config_dict(v):
    # for vLLM specific
    vlist = v.split(", ")
    ret = {}
    for e in vlist:
        sl = e.split(": ")
        if sl[0] in __skip_config_args__:
            continue
        ret[sl[0]] = int(sl[1])
    return ret


def translate_dejavu_cache(cache_path):
    print(f"Exporting {cache_path} to {__vllm_base_path__}...")
    # tag_path = os.path.dirname(cache_path)
    # gpu_name_path = os.path.dirname(os.path.dirname(tag_path[:-2])[:-2])
    # gpu_name = os.path.basename(gpu_name_path)[4:]
    # adapt to new structure
    path_ids = os.path.abspath(cache_path).split("/")
    gpu_name_path = path_ids[-7]
    gpu_name = gpu_name_path[4:]

    with open(cache_path, "r") as f:
        dejavu_cache = json.load(f)

    cache_dict = dejavu_cache["cache"]

    # k0 = list(cache_dict.keys())[0]
    # v0 = cache_dict[k0]
    num_experts = None

    config_per_device = {}
    timings_per_device = {}
    for k, v in cache_dict.items():
        kd = moe_key_to_param_dict(k)
        vd = create_config_dict(v)
        ot = dejavu_cache["timings"][k]["values"][
            dejavu_cache["timings"][k]["lables"].index("ms")
        ]
        if num_experts is None:
            num_experts = int(kd["E"][1:-1])
        else:
            assert num_experts == int(kd["E"][1:-1])
        # num_tokens = int(kd['num_valid_tokens'][1:-1])
        # TODO: how to automatically determine /2? update method signature?
        # num_tokens = int(int(kd['num_valid_tokens'][1:-1]) / 2)
        num_tokens = int(kd["num_actual_tokens"][1:-1])
        # N = int(kd['N'][1:-1])/num_tokens
        # N = int(kd['N'][1:-1])
        # vllm_N = int(kd['stride_am'][1:-1])
        vllm_N = int(kd["K"][1:-1])
        # N = int(kd['stride_am'][1:-1])/2  # due to test script shape generation?
        new_dict = {num_tokens: vd}
        if vllm_N not in config_per_device:
            config_per_device[vllm_N] = new_dict
            timings_per_device[vllm_N] = {num_tokens: ot}
        else:
            # config_per_device[vllm_N].update(new_dict)
            if num_tokens not in config_per_device[vllm_N]:
                config_per_device[vllm_N][num_tokens] = vd
                timings_per_device[vllm_N][num_tokens] = ot
            else:
                if ot >= timings_per_device[vllm_N][num_tokens]:
                    print(
                        f"configuration for {num_tokens} already existent: {config_per_device[vllm_N][num_tokens]}; "
                        f"would overwrite with {vd} but is SLOWER, skipping..."
                    )
                else:
                    print(
                        f"configuration for {num_tokens} already existent: {config_per_device[vllm_N][num_tokens]}; "
                        f"overwrite with {vd} because it is FASTER..."
                    )
                    config_per_device[vllm_N][num_tokens] = vd
                    timings_per_device[vllm_N][num_tokens] = ot

    modified_paths = []
    for N, config_dict in config_per_device.items():
        file_name = f"E={int(num_experts)},N={int(N)},device_name={gpu_name}.json"
        modified_paths.append(file_name)
        target_path = os.path.abspath(f"{__vllm_base_path__}/{file_name}")
        # num_tokens / M as key in dict
        with open(target_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    print(f"modified the following files: {modified_paths}")
    print(f"triton-dejavu has saved {dejavu_cache['total_bench_time_s']}s")
    print("...done")


if __name__ == "__main__":
    translate_dejavu_cache(sys.argv[1])
