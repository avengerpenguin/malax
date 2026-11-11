"""The single MLX thread and the one model it holds; shared by the mflux and mlx-vlm runners."""

import asyncio
import gc
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import mlx.core as mx


def configure_mlx(cache_limit_gb: float) -> None:
    # MLX keeps freed buffers for reuse; uncapped, that cache can be as big as the model itself.
    mx.set_cache_limit(int(cache_limit_gb * 1024**3))


async def run_on_mlx_thread(executor: ThreadPoolExecutor, func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


def clear_mlx_cache_sync():
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def load_sync(loader: Callable[[], Any]) -> Any:
    model = loader()
    # Free intermediates from loading (e.g. LoRA baking) before generation needs the memory.
    gc.collect()
    mx.clear_cache()
    return model


def memory_stats() -> str:
    gb = 1024**3
    return (
        f"active={mx.get_active_memory() / gb:.1f}GB "
        f"cache={mx.get_cache_memory() / gb:.1f}GB "
        f"peak={mx.get_peak_memory() / gb:.1f}GB"
    )


class ActiveModel:
    """Holds at most one model (chat or image) in memory at a time."""

    def __init__(self, executor: ThreadPoolExecutor):
        # MLX work must stay on this one thread; the lock below serialises use of it.
        self.executor = executor
        self.name: str | None = None
        self.model: Any | None = None
        self.lock = asyncio.Lock()
        self.waiting_requests: int = 0

    async def run(self, func, *args, **kwargs):
        return await run_on_mlx_thread(self.executor, func, *args, **kwargs)

    @asynccontextmanager
    async def session(self):
        self.waiting_requests += 1
        try:
            await self.lock.acquire()
        finally:
            self.waiting_requests -= 1

        try:
            yield
        finally:
            self.lock.release()

    async def load(self, name: str, loader: Callable[[], Any]) -> Any:
        if self.name == name:
            return self.model

        await self.unload()

        print(f"Loading model: {name}")
        self.model = await self.run(load_sync, loader)
        self.name = name
        print(f"Loaded model: {name} ({memory_stats()})")

        return self.model

    async def unload(self):
        if self.model is None:
            return

        print(f"Unloading model: {self.name} ({memory_stats()})")

        # Chat models are a (model, processor, config) tuple; track the model itself.
        target = self.model[0] if isinstance(self.model, tuple) else self.model
        try:
            ref = weakref.ref(target)
        except TypeError:
            ref = None
        del target

        self.model = None
        self.name = None

        await self.run(clear_mlx_cache_sync)

        if ref is not None and (leaked := ref()) is not None:
            holders = [type(r).__name__ for r in gc.get_referrers(leaked)]
            print(f"WARNING: model still alive after unload, referenced by: {holders}")
            del leaked
        print(f"Unloaded ({memory_stats()})")
