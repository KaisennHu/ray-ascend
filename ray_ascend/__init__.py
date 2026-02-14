"""Ray Ascend - Ray extensions for Huawei Ascend NPU.

This package provides:
- HCCL collective communication among ray actors
- Transport Ascend NPU tensors via HCCS
- Transport Ascend NPU tensors via HCCS and CPU tensors via RDMA
"""

from .collective import HCCLGroup
from .direct_transport import HCCLTensorTransport, YRTensorTransport

__all__ = [
    "HCCLGroup",
    "HCCLTensorTransport",
    "YRTensorTransport",
    "register_hccl_collective_backend",
    "register_hccl_tensor_transport",
    "register_yr_tensor_transport",
]


def register_hccl_collective_backend() -> None:
    """
    Register HCCL collective backend for Ray.

    This function must be called in each Ray worker/actor process
    before using HCCL collective operations.

    Example:
        from ray.util import collective
        from ray_ascend import register_hccl_collective_backend

        register_hccl_collective_backend()

        @ray.remote(resources={"NPU": 1})
        class RayActor:
            def __init__(self):
                register_hccl_collective_backend()

        collective.create_collective_group(
            actors,
            len(actors),
            list(range(0, len(actors))),
            backend="HCCL",
            group_name="my_group",
        )
    """
    from ray.util.collective.backend_registry import register_collective_backend

    register_collective_backend("HCCL", HCCLGroup)


def register_hccl_tensor_transport() -> None:
    """
    Register HCCL backend and tensor transport for Ray.

    This function must be called in each Ray worker/actor process
    before using HCCL collective operations or tensor transport.

    Example:
        from ray_ascend import register_hccl_tensor_transport

        register_hccl_tensor_transport()

        @ray.remote(resources={"NPU": 1})
        class RayActor:
            def __init__(self):
                register_hccl_tensor_transport()

            @ray.method(tensor_transport="HCCL")
            def transfer_npu_tensor_via_hccs(self):
                return torch.tensor([1, 2, 3]).npu()
    """
    from ray.experimental import register_tensor_transport

    register_hccl_collective_backend()
    register_tensor_transport("HCCL", ["npu"], HCCLTensorTransport)


def register_yr_tensor_transport(devices=None) -> None:
    """
    Register YR tensor transport for Ray.

    This function must be called in each Ray worker/actor process
    before using YR tensor transport for efficient NPU-to-NPU transfers.

    Args:
        devices: List of device types to support. Can be:
            - ["npu"] for NPU tensors only
            - ["npu", "cpu"] for NPU and CPU tensors
            - ["cpu"] for CPU tensors only
            - None (default) for both ["npu", "cpu"]

    Example:
        from ray_ascend import register_yr_tensor_transport

        register_yr_tensor_transport(["npu", "cpu"])

        class RayActor:
            def __init__(self):
                register_yr_tensor_transport(["npu", "cpu"])

            @ray.method(tensor_transport="YR")
            def transfer_npu_tensor_via_hccs():
                return torch.zeros(1024, device="npu")

            @ray.method(tensor_transport="YR")
            def transfer_cpu_tensor_via_rdma():
                return torch.zeros(1024)
    """
    from ray.experimental import register_tensor_transport

    if devices is None:
        devices = ["npu", "cpu"]

    register_tensor_transport("YR", devices, YRTensorTransport)
