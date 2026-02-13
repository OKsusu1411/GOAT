from __future__ import annotations
from lib.utils.wrapper_utils import *
from lib.utils.Running_mean_std import RunningMeanStd

import logging
import os
import sys
import random
import time

import numpy as np


__all__ = ["__version__", "logger", "config"]


# read library version from metadata
try:
    import importlib.metadata

    __version__ = importlib.metadata.version("skrl")
except ImportError:
    __version__ = "unknown"


# logger with format
class _Formatter(logging.Formatter):
    _format = "[%(name)s:%(levelname)s] %(message)s"
    _formats = {
        logging.DEBUG: f"\x1b[38;20m{_format}\x1b[0m",
        logging.INFO: f"\x1b[38;20m{_format}\x1b[0m",
        logging.WARNING: f"\x1b[33;20m{_format}\x1b[0m",
        logging.ERROR: f"\x1b[31;20m{_format}\x1b[0m",
        logging.CRITICAL: f"\x1b[31;1m{_format}\x1b[0m",
    }

    def format(self, record):
        return logging.Formatter(self._formats.get(record.levelno)).format(record)


_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(_Formatter())

logger = logging.getLogger("skrl")
logger.setLevel(logging.DEBUG)
logger.addHandler(_handler)


# machine learning framework configuration
class _Config(object):
    def __init__(self) -> None:
        """Machine learning framework specific configuration."""

        class PyTorch(object):
            def __init__(self) -> None:
                """PyTorch configuration."""
                self._key = 0
                # torch.distributed config
                self._local_rank = int(os.getenv("LOCAL_RANK", "0"))
                self._rank = int(os.getenv("RANK", "0"))
                self._world_size = int(os.getenv("WORLD_SIZE", "1"))
                self._is_distributed = self._world_size > 1
                # device
                self._device = f"cuda:{self._local_rank}"

                # set up distributed runs
                if self._is_distributed:
                    import torch

                    logger.info(
                        f"Distributed (rank: {self._rank}, local rank: {self._local_rank}, world size: {self._world_size})"
                    )
                    torch.distributed.init_process_group("nccl", rank=self._rank, world_size=self._world_size)
                    torch.cuda.set_device(self._local_rank)

            @staticmethod
            def parse_device(device: str | "torch.device" | None, validate: bool = True) -> "torch.device":
                """Parse the input device and return a :py:class:`~torch.device` instance.

                :param device: Device specification. If the specified device is ``None`` or it cannot be resolved,
                    the default available device will be returned instead.
                :param validate: Whether to check that the specified device is valid. Since PyTorch does not check if
                    the specified device index is valid, a tensor is created for the verification.

                :return: PyTorch device.
                """
                import torch

                _device = None
                if isinstance(device, torch.device):
                    _device = device
                elif isinstance(device, str):
                    try:
                        _device = torch.device(device)
                    except RuntimeError as e:
                        logger.warning(f"Invalid device specification ({device}): {e}")
                if _device is None:
                    _device = torch.device(
                        "cuda:0" if torch.cuda.is_available() else "cpu"
                    )  # torch.get_default_device() was introduced in version 2.3.0
                # validate device
                if validate:
                    try:
                        torch.zeros((1,), device=_device)
                    except Exception as e:
                        logger.warning(f"Invalid device specification ({device}): {e}")
                        _device = PyTorch.parse_device(None)
                return _device

            @property
            def device(self) -> "torch.device":
                """Default device.

                The default device, unless specified, is ``cuda:0`` (or ``cuda:LOCAL_RANK`` in a distributed environment)
                if CUDA is available, ``cpu`` otherwise.
                """
                self._device = self.parse_device(self._device, validate=False)
                return self._device

            @device.setter
            def device(self, device: str | "torch.device") -> None:
                self._device = device

            @property
            def key(self) -> int:
                """Pseudo-random number generator (PRNG) key."""
                return self._key

            @key.setter
            def key(self, value: int) -> None:
                self._key = value

            @property
            def local_rank(self) -> int:
                """The rank of the worker/process (e.g.: GPU) within a local worker group (e.g.: node).

                This property reads from the ``LOCAL_RANK`` environment variable (``0`` if it doesn't exist).

                Read-only attribute.
                """
                return self._local_rank

            @property
            def rank(self) -> int:
                """The rank of the worker/process (e.g.: GPU) within a worker group (e.g.: across all nodes).

                This property reads from the ``RANK`` environment variable (``0`` if it doesn't exist).

                Read-only attribute.
                """
                return self._rank

            @property
            def world_size(self) -> int:
                """The total number of workers/process (e.g.: GPUs) in a worker group (e.g.: across all nodes).

                This property reads from the ``WORLD_SIZE`` environment variable (``1`` if it doesn't exist).

                Read-only attribute.
                """
                return self._world_size

            @property
            def is_distributed(self) -> bool:
                """Whether if running in a distributed environment.

                This property is ``True`` when the PyTorch's distributed environment variable ``WORLD_SIZE > 1``.

                Read-only attribute.
                """
                return self._is_distributed

        class JAX(object):
            def __init__(self) -> None:
                """JAX configuration."""
                self._key = np.array([0, 0], dtype=np.uint32)
                # distributed config (based on torch.distributed, since JAX doesn't implement it)
                # JAX doesn't automatically start multiple processes from a single program invocation
                # https://jax.readthedocs.io/en/latest/multi_process.html#launching-jax-processes
                self._local_rank = int(os.getenv("JAX_LOCAL_RANK", "0"))
                self._rank = int(os.getenv("JAX_RANK", "0"))
                self._world_size = int(os.getenv("JAX_WORLD_SIZE", "1"))
                self._coordinator_address = (
                    os.getenv("JAX_COORDINATOR_ADDR", "127.0.0.1") + ":" + os.getenv("JAX_COORDINATOR_PORT", "1234")
                )
                self._is_distributed = self._world_size > 1
                # device
                self._device = f"cuda:{self._local_rank}"

                # set up distributed runs
                if self._is_distributed:
                    import jax

                    logger.info(
                        f"Distributed (rank: {self._rank}, local rank: {self._local_rank}, world size: {self._world_size})"
                    )
                    jax.distributed.initialize(
                        coordinator_address=self._coordinator_address,
                        num_processes=self._world_size,
                        process_id=self._rank,
                        local_device_ids=self._local_rank,
                    )
                    # get the device local to process
                    try:
                        self._device = jax.local_devices(process_index=self._rank)[0]
                        logger.info(f"Using device local to process with index/rank {self._rank} ({self._device})")
                    except Exception as e:
                        logger.warning(f"Failed to get the device local to process with index/rank {self._rank}: {e}")

            @staticmethod
            def parse_device(device: str | "jax.Device" | None) -> "jax.Device":
                """Parse the input device and return a :py:class:`~jax.Device` instance.

                .. hint::

                    This function supports the PyTorch-like ``"type:ordinal"`` string specification (e.g.: ``"cuda:0"``).

                .. warning::

                    This method returns (forces to use) the device local to process in a distributed environment.

                :param device: Device specification. If the specified device is ``None`` or it cannot be resolved,
                    the default available device will be returned instead.

                :return: JAX Device.
                """
                import jax

                # force the use of the device local to process in distributed runs
                if config.jax.is_distributed:
                    try:
                        return jax.local_devices(process_index=config.jax.rank)[0]
                    except Exception as e:
                        logger.warning(
                            f"Failed to get the device local to process with index/rank {config.jax.rank}: {e}"
                        )

                if isinstance(device, jax.Device):
                    return device
                elif isinstance(device, str):
                    device_type, device_index = f"{device}:0".split(":")[:2]
                    try:
                        return jax.devices(device_type)[int(device_index)]
                    except (RuntimeError, IndexError) as e:
                        logger.warning(f"Invalid device specification ({device}): {e}")
                return jax.devices()[0]

            @property
            def device(self) -> "jax.Device":
                """Default device.

                The default device, unless specified, is ``cuda:0`` if CUDA is available, ``cpu`` otherwise.
                However, in a distributed environment, it is the device local to process with index ``JAX_RANK``.
                """
                self._device = self.parse_device(self._device)
                return self._device

            @device.setter
            def device(self, device: str | "jax.Device") -> None:
                self._device = device
                if not isinstance(self._key, np.ndarray):
                    import jax

                    self._key = np.asarray(jax.device_get(self._key))

            @property
            def key(self) -> "jax.Array":
                """Pseudo-random number generator (PRNG) key.

                Key is formatted as 32-bit unsigned integer and the default device is used.
                """
                if isinstance(self._key, np.ndarray):
                    try:
                        import jax

                        with jax.default_device(self.device):
                            self._key = jax.random.PRNGKey(self._key[1])
                    except ImportError:
                        pass
                return self._key

            @key.setter
            def key(self, value: int | np.ndarray | "jax.Array") -> None:
                if isinstance(value, (int, float)):
                    value = np.array([0, value], dtype=np.uint32)
                self._key = value

            @property
            def local_rank(self) -> int:
                """The rank of the worker/process (e.g.: GPU) within a local worker group (e.g.: node).

                This property reads from the ``JAX_LOCAL_RANK`` environment variable (``0`` if it doesn't exist).

                Read-only attribute.
                """
                return self._local_rank

            @property
            def rank(self) -> int:
                """The rank of the worker/process (e.g.: GPU) within a worker group (e.g.: across all nodes).

                This property reads from the ``JAX_RANK`` environment variable (``0`` if it doesn't exist).

                Read-only attribute.
                """
                return self._rank

            @property
            def world_size(self) -> int:
                """The total number of workers/process (e.g.: GPUs) in a worker group (e.g.: across all nodes).

                This property reads from the ``JAX_WORLD_SIZE`` environment variable (``1`` if it doesn't exist).

                Read-only attribute.
                """
                return self._world_size

            @property
            def coordinator_address(self) -> int:
                """IP address and port where process 0 will start a JAX service.

                This property reads from the ``JAX_COORDINATOR_ADDR:JAX_COORDINATOR_PORT`` environment variables
                (``127.0.0.1:1234`` if they don't exist).

                Read-only attribute.
                """
                return self._coordinator_address

            @property
            def is_distributed(self) -> bool:
                """Whether if running in a distributed environment.

                This property is ``True`` when the JAX's distributed environment variable ``JAX_WORLD_SIZE > 1``.

                Read-only attribute.
                """
                return self._is_distributed

        class Warp(object):
            def __init__(self) -> None:
                """Warp configuration."""
                self._key = 0
                # device
                self._device = "cuda:0"
                # kernel-related config
                self.tiled = True
                self.block_dim = 128
                self.tile_dim_0 = 32
                self.tile_dim_1 = 32
                self.tile_dim_2 = 32

                # init Warp (don't import if it hasn't been imported)
                if "warp" in sys.modules:
                    import warp as wp

                    wp.init()

            @staticmethod
            def parse_device(device: str | "warp.context.Device" | None) -> "warp.context.Device":
                """Parse the input device and return a :py:class:`~warp.context.Device` instance.

                :param device: Device specification. If the specified device is ``None`` or it cannot be resolved,
                    the default available device will be returned instead.

                :return: Warp Device.
                """
                import warp as wp

                if isinstance(device, wp.context.Device):
                    return device
                elif isinstance(device, str):
                    try:
                        return wp.get_device(device)
                    except ValueError as e:
                        logger.warning(f"Invalid device specification ({device}): {e}")
                return wp.get_device()

            @property
            def device(self) -> "warp.context.Device":
                """Default device.

                The default device, unless specified, is ``cuda`` if CUDA is available, ``cpu`` otherwise.
                """
                self._device = self.parse_device(self._device)
                return self._device

            @device.setter
            def device(self, device: str | "warp.context.Device") -> None:
                self._device = device

            @property
            def key(self) -> int:
                """Pseudo-random number generator (PRNG) key."""
                return self._key

            @key.setter
            def key(self, value: int) -> None:
                self._key = value

        self.jax = JAX()
        self.warp = Warp()
        self.torch = PyTorch()


config = _Config()

def set_seed(seed: int | None = None, deterministic: bool = False) -> int:
    """Set the seed for the random number generators.

    .. note::

        In distributed runs, the worker/process seed will be incremented (counting from the defined value)
        according to its rank.

    .. warning::

        Due to NumPy's legacy seeding constraint the seed must be between 0 and 2**32 - 1.
        Otherwise a NumPy exception (``ValueError: Seed must be between 0 and 2**32 - 1``) will be raised.

    Modified packages:

    - ``random``
    - ``numpy``
    - ``torch`` (if available)
    - ``skrl`` (PRNG keys: ``config.torch.key``, ``config.jax.key``, ``config.warp.key``)

    Example:

    .. code-block:: python

        # fixed seed
        >>> from skrl.utils import set_seed
        >>> set_seed(42)
        [skrl:INFO] Seed: 42
        42

        # random seed
        >>> from skrl.utils import set_seed
        >>> set_seed()
        [skrl:INFO] Seed: 1776118066
        1776118066

        # enable deterministic. The following environment variables should be established:
        # - CUDA 10.1: CUDA_LAUNCH_BLOCKING=1
        # - CUDA 10.2 or later: CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:8
        >>> from skrl.utils import set_seed
        >>> set_seed(42, deterministic=True)
        [skrl:INFO] Seed: 42
        [skrl:WARNING] PyTorch/cuDNN deterministic algorithms are enabled. This may affect performance
        42

    :param seed: The seed to set. If ``None``, a random seed will be generated.
    :param deterministic: Whether PyTorch is configured to use deterministic algorithms.
        The following environment variables should be established for CUDA 10.1 (``CUDA_LAUNCH_BLOCKING=1``)
        and for CUDA 10.2 or later (``CUBLAS_WORKSPACE_CONFIG=:16:8`` or ``CUBLAS_WORKSPACE_CONFIG=:4096:8``).
        See PyTorch `Reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`_ for details.

    :return: Seed.
    """
    # generate a random seed
    if seed is None:
        try:
            seed = int.from_bytes(os.urandom(4), byteorder=sys.byteorder)
        except NotImplementedError:
            seed = int(time.time() * 1000)
        seed %= 2**31  # NumPy's legacy seeding seed must be between 0 and 2**32 - 1
    seed = int(seed)

    # set different seeds in distributed runs
    if config.torch.is_distributed:
        seed += config.torch.rank
    if config.jax.is_distributed:
        seed += config.jax.rank

    logger.info(f"Seed: {seed}")

    # python / numpy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # torch
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # On CUDA 10.1, set environment variable CUDA_LAUNCH_BLOCKING=1
            # On CUDA 10.2 or later, set environment variable CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:8
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            logger.warning("PyTorch/cuDNN deterministic algorithms are enabled. This may affect performance")
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"PyTorch seeding error: {e}")

    # framework PRNG key
    config.torch.key = seed
    config.jax.key = seed
    config.warp.key = seed

    return seed


class ScopedTimer:
    """Scoped timer that can be used to time the execution of a block of code."""

    def __enter__(self):
        self._elapsed_time = None
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._elapsed_time = time.time() - self._start_time

    @property
    def elapsed_time(self) -> float:
        """Elapsed time (in seconds).

        .. note::

            If called within the scope of the context manager, the elapsed time is updated to reflect the time
            spent within the scope. If called outside the context manager scope, the elapsed time is fixed to
            the time at which the context manager was exited.

        :return: Elapsed time in seconds.
        """
        if self._elapsed_time is None:
            return time.time() - self._start_time
        return self._elapsed_time

    @property
    def elapsed_time_ms(self) -> float:
        """Elapsed time (in milliseconds).

        .. note::

            If called within the scope of the context manager, the elapsed time is updated to reflect the time
            spent within the scope. If called outside the context manager scope, the elapsed time is fixed to
            the time at which the context manager was exited.

        :return: Elapsed time in milliseconds.
        """
        if self._elapsed_time is None:
            return (time.time() - self._start_time) * 1000
        return self._elapsed_time * 1000