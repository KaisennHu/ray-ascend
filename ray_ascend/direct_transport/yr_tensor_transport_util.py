import pickle
import warnings

from yr.datasystem import KVClient

try:
    import torch_npu
    from yr.datasystem import DsTensorClient
except ImportError:
    warnings.warn(
        "The 'yr_tensor_transport' feature requires optional dependencies "
        "(yr, torch_npu). CPU-only paths can still work, but NPU transport "
        "will be unavailable. Install with: pip install 'ray-ascend[yr]'",
        RuntimeWarning,
    )


from abc import ABC, abstractmethod


def raise_if_failed(failed_keys, action):
    if failed_keys:
        raise RuntimeError(f"Failed to {action} keys: {failed_keys}")


class BaseDSAdapter(ABC):
    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def put(self, keys, tensors):
        pass

    @abstractmethod
    def get(self, keys, tensors):
        pass

    @abstractmethod
    def delete(self, keys):
        pass


class CPUClientAdapter(BaseDSAdapter):
    def __init__(self, host, port):
        self._client = KVClient(host=host, port=port)

    def init(self):
        self._client.init()

    def put(self, keys, tensors):
        # TODO: Do zero-copy optimization laster.
        values = [pickle.dumps(t) for t in tensors]
        failed_keys = self._client.mset(keys=keys, vals=values)
        raise_if_failed(failed_keys, "put")

    def get(self, keys, tensors):
        raw_tensors = self._client.get(keys=keys)
        tensors[:] = [pickle.loads(r) for r in raw_tensors]

    def delete(self, keys):
        failed_keys = self._client.delete(keys=keys)
        raise_if_failed(failed_keys, "delete")


class NPUClientAdapter(BaseDSAdapter):
    def __init__(self, host, port):
        self._client = DsTensorClient(
            host=host,
            port=port,
            device_id=0,
            connect_timeout_ms=60000,
        )

    def init(self):
        self._client.init()

    def put(self, keys, tensors):
        failed_keys = self._client.dev_mset(keys=keys, tensors=tensors)
        raise_if_failed(failed_keys, "put")

    def get(self, keys, tensors):
        failed_keys = self._client.dev_mget(keys=keys, tensors=tensors)
        raise_if_failed(failed_keys, "get")

    def delete(self, keys):
        failed_keys = self._client.dev_delete(keys=keys)
        raise_if_failed(failed_keys, "delete")
