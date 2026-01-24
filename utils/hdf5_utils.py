# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
HDF5 utilities for efficient feature storage and retrieval.
Inspired by preprocess_semantic_base.py implementation.
"""

import os
import json
import h5py
import numpy as np
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor


class HDF5Writer:
    """Manages HDF5 file writing with multi-file support for vocoder features."""

    def __init__(
        self,
        base_dir: str,
        samples_per_file: int = 10000,
        use_float16: bool = False,
        compression: str = "gzip",
        compression_opts: int = 4
    ):
        """
        Initialize HDF5 writer.

        Args:
            base_dir: Directory to store HDF5 files
            samples_per_file: Maximum samples per HDF5 file
            use_float16: Whether to convert float32 to float16 for storage
            compression: Compression algorithm (gzip, lzf, or None)
            compression_opts: Compression level (1-9 for gzip)
        """
        self.base_dir = Path(base_dir)
        self.samples_per_file = samples_per_file
        self.use_float16 = use_float16
        self.compression = compression
        self.compression_opts = compression_opts
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Mapping from sample key to HDF5 file location
        self.sample_index = {}  # {sample_key: {"file_idx": int, "group_name": str}}
        self.current_file_idx = 0
        self.current_file_count = 0
        self.current_file = None
        self.lock = threading.Lock()

        # Index file path
        self.index_path = self.base_dir / "hdf5_index.json"
        self.load_index()

    def load_index(self):
        """Load existing HDF5 index if available."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                data = json.load(f)
                self.sample_index = data.get("sample_index", {})
                self.current_file_idx = data.get("current_file_idx", 0)
                self.current_file_count = data.get("current_file_count", 0)

    def save_index(self):
        """Save HDF5 index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump({
                "sample_index": self.sample_index,
                "current_file_idx": self.current_file_idx,
                "current_file_count": self.current_file_count,
                "samples_per_file": self.samples_per_file,
                "use_float16": self.use_float16
            }, f, indent=2)

    def get_hdf5_path(self, file_idx: int) -> Path:
        """Get HDF5 file path for a given file index."""
        return self.base_dir / f"features_{file_idx:05d}.h5"

    def sample_exists(self, sample_key: str) -> bool:
        """Check if a sample already exists in the index."""
        return sample_key in self.sample_index

    def write_sample(self, sample_key: str, data: Dict[str, np.ndarray]) -> bool:
        """
        Write a single sample to HDF5 file.

        Args:
            sample_key: Unique identifier for the sample (e.g., uid)
            data: Dictionary of feature name to numpy array

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                # Check if we need to create a new file
                if self.current_file_count >= self.samples_per_file or self.current_file is None:
                    if self.current_file is not None:
                        self.current_file.close()

                    # Increment file index
                    if self.current_file is not None or self.current_file_count > 0:
                        self.current_file_idx += 1
                        self.current_file_count = 0
                    elif self.current_file_idx == 0:
                        self.current_file_idx = 1
                        self.current_file_count = 0

                    hdf5_path = self.get_hdf5_path(self.current_file_idx)
                    self.current_file = h5py.File(hdf5_path, 'a')

                # Create group for this sample
                if sample_key in self.current_file:
                    del self.current_file[sample_key]

                sample_group = self.current_file.create_group(sample_key)

                # Write each dataset
                for key, value in data.items():
                    # Convert to float16 if requested and data is floating point
                    if self.use_float16 and np.issubdtype(value.dtype, np.floating):
                        value = value.astype(np.float16)

                    if self.compression:
                        sample_group.create_dataset(
                            key,
                            data=value,
                            compression=self.compression,
                            compression_opts=self.compression_opts
                        )
                    else:
                        sample_group.create_dataset(key, data=value)

                # Update index
                self.sample_index[sample_key] = {
                    "file_idx": self.current_file_idx,
                    "group_name": sample_key
                }

                self.current_file_count += 1

                # Periodically save index
                if self.current_file_count % 100 == 0:
                    self.save_index()

                return True
            except Exception as e:
                print(f"Failed to write sample {sample_key}: {e}")
                return False

    def close(self):
        """Close current HDF5 file and save index."""
        with self.lock:
            if self.current_file is not None:
                self.current_file.close()
                self.current_file = None
            self.save_index()


class HDF5Reader:
    """Manages HDF5 file reading with caching for vocoder features."""

    def __init__(self, base_dir: str, cache_size: int = 5):
        """
        Initialize HDF5 reader.

        Args:
            base_dir: Directory containing HDF5 files
            cache_size: Number of HDF5 files to keep open in cache
        """
        self.base_dir = Path(base_dir)
        self.cache_size = cache_size
        self.index_path = self.base_dir / "hdf5_index.json"

        # Load index
        self.sample_index = {}
        self.load_index()

        # File handle cache (LRU-like)
        self.file_cache = {}  # {file_idx: h5py.File}
        self.file_access_order = []  # Track access order for LRU
        self.lock = threading.Lock()

    def load_index(self):
        """Load HDF5 index from disk."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                data = json.load(f)
                self.sample_index = data.get("sample_index", {})
        else:
            raise FileNotFoundError(f"HDF5 index not found at {self.index_path}")

    def get_hdf5_path(self, file_idx: int) -> Path:
        """Get HDF5 file path for a given file index."""
        return self.base_dir / f"features_{file_idx:05d}.h5"

    def _get_file_handle(self, file_idx: int) -> h5py.File:
        """Get file handle with caching."""
        with self.lock:
            if file_idx in self.file_cache:
                # Move to end of access order (most recently used)
                if file_idx in self.file_access_order:
                    self.file_access_order.remove(file_idx)
                self.file_access_order.append(file_idx)
                return self.file_cache[file_idx]

            # Open new file
            hdf5_path = self.get_hdf5_path(file_idx)
            file_handle = h5py.File(hdf5_path, 'r')

            # Add to cache
            self.file_cache[file_idx] = file_handle
            self.file_access_order.append(file_idx)

            # Evict oldest if cache is full
            while len(self.file_cache) > self.cache_size:
                oldest_idx = self.file_access_order.pop(0)
                if oldest_idx in self.file_cache:
                    self.file_cache[oldest_idx].close()
                    del self.file_cache[oldest_idx]

            return file_handle

    def read_sample(self, sample_key: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Read a single sample from HDF5 file.

        Args:
            sample_key: Unique identifier for the sample

        Returns:
            Dictionary of feature name to numpy array, or None if not found
        """
        if sample_key not in self.sample_index:
            return None

        try:
            info = self.sample_index[sample_key]
            file_idx = info["file_idx"]
            group_name = info["group_name"]

            file_handle = self._get_file_handle(file_idx)

            if group_name not in file_handle:
                return None

            sample_group = file_handle[group_name]
            data = {}
            for key in sample_group.keys():
                data[key] = sample_group[key][:]

            return data
        except Exception as e:
            print(f"Failed to read sample {sample_key}: {e}")
            return None

    def read_feature(self, sample_key: str, feature_name: str) -> Optional[np.ndarray]:
        """
        Read a specific feature from a sample.

        Args:
            sample_key: Unique identifier for the sample
            feature_name: Name of the feature to read

        Returns:
            Numpy array or None if not found
        """
        if sample_key not in self.sample_index:
            return None

        try:
            info = self.sample_index[sample_key]
            file_idx = info["file_idx"]
            group_name = info["group_name"]

            file_handle = self._get_file_handle(file_idx)

            if group_name not in file_handle:
                return None

            sample_group = file_handle[group_name]

            if feature_name not in sample_group:
                return None

            return sample_group[feature_name][:]
        except Exception as e:
            print(f"Failed to read feature {feature_name} from {sample_key}: {e}")
            return None

    def get_all_sample_keys(self) -> List[str]:
        """Get all sample keys in the index."""
        return list(self.sample_index.keys())

    def close(self):
        """Close all cached file handles."""
        with self.lock:
            for file_handle in self.file_cache.values():
                file_handle.close()
            self.file_cache.clear()
            self.file_access_order.clear()

    def __del__(self):
        """Destructor to ensure files are closed."""
        self.close()


class AsyncHDF5Saver:
    """Handles asynchronous HDF5 saving with thread pool."""

    def __init__(self, hdf5_writer: HDF5Writer, num_workers: int = 4):
        """
        Initialize async saver.

        Args:
            hdf5_writer: HDF5Writer instance
            num_workers: Number of worker threads
        """
        self.hdf5_writer = hdf5_writer
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def save_async(self, sample_key: str, data: Dict[str, np.ndarray]):
        """Submit save task to thread pool."""
        return self.executor.submit(self.hdf5_writer.write_sample, sample_key, data)

    def shutdown(self):
        """Shutdown the executor and close HDF5 writer."""
        self.executor.shutdown(wait=True)
        self.hdf5_writer.close()
