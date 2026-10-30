"""Low-level HDF5 helpers that copy objects faithfully between files.

The goal is for the subset product to be indistinguishable in *layout* from a
full-size product: same groups, link order, datatypes (including the NISAR
``complex32`` compound type), fill values, chunking and compression filters,
attributes and HDF5 dimension scales. h5py's high-level API drops some of
these properties, so dataset and group creation goes through the low-level
``h5py.h5d`` / ``h5py.h5g`` interfaces using copies of the source creation
property lists.
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

import h5py
import numpy as np

log = logging.getLogger(__name__)

# Attributes maintained by the HDF5 dimension-scale API. They hold object
# references into the *source* file, so they are never copied verbatim;
# scales are re-created and re-attached in the output (see `copy_dimension_scales`).
_DIMSCALE_ATTRS = {"DIMENSION_LIST", "REFERENCE_LIST"}

# Target size of one block when streaming dataset contents.
_BLOCK_BYTES = 64 * 1024 * 1024


def _dtype_has_reference(dt: np.dtype) -> bool:
    """True if `dt` is, or contains, an HDF5 object/region reference."""
    if h5py.check_dtype(ref=dt) is not None:
        return True
    base = h5py.check_dtype(vlen=dt)
    if isinstance(base, np.dtype) and _dtype_has_reference(base):
        return True
    if dt.fields:
        return any(_dtype_has_reference(f[0]) for f in dt.fields.values())
    if dt.subdtype is not None:
        return _dtype_has_reference(dt.subdtype[0])
    return False


def is_dimension_scale(obj) -> bool:
    return isinstance(obj, h5py.Dataset) and h5py.h5ds.is_scale(obj.id)


def copy_attrs(src, dst, skip: Iterable[str] = ()) -> None:
    """Copy all attributes of `src` onto `dst`, preserving type and shape.

    Dimension-scale bookkeeping attributes and any attribute holding object
    references are skipped (references would point into the source file).
    """
    skip = set(skip) | _DIMSCALE_ATTRS
    scale = is_dimension_scale(src)
    for name in src.attrs:
        if name in skip:
            continue
        if scale and name in ("CLASS", "NAME"):
            # Re-created by make_scale() in copy_dimension_scales().
            continue
        aid = src.attrs.get_id(name)
        if _dtype_has_reference(aid.dtype):
            log.debug("Skipping reference attribute %s@%s", src.name, name)
            continue
        try:
            _copy_attr_lowlevel(src, dst, name)
        except Exception:  # pragma: no cover - fallback path
            dst.attrs.create(name, data=src.attrs[name], shape=aid.shape, dtype=aid.dtype)


def _copy_attr_lowlevel(src, dst, name: str) -> None:
    """Copy one attribute byte-for-byte (same HDF5 type and dataspace)."""
    bname = name.encode()
    aid = h5py.h5a.open(src.id, bname)
    tid = aid.get_type()
    sid = aid.get_space()
    if _dtype_has_vlen(aid.dtype):
        # Variable-length data: let h5py handle the memory conversion.
        dst.attrs.create(name, data=src.attrs[name], shape=aid.shape, dtype=aid.dtype)
        return
    if h5py.h5a.exists(dst.id, bname):
        h5py.h5a.delete(dst.id, bname)
    new = h5py.h5a.create(dst.id, bname, tid, sid)
    if sid.get_simple_extent_type() == h5py.h5s.NULL:
        return
    arr = np.empty(aid.shape, dtype=aid.dtype)
    aid.read(arr)
    new.write(arr)


def _dtype_has_vlen(dt: np.dtype) -> bool:
    if h5py.check_dtype(vlen=dt) is not None or dt.kind == "O":
        return True
    if dt.fields:
        return any(_dtype_has_vlen(f[0]) for f in dt.fields.values())
    if dt.subdtype is not None:
        return _dtype_has_vlen(dt.subdtype[0])
    return False


def create_group_like(src: h5py.Group, parent: h5py.Group, name: str) -> h5py.Group:
    """Create `parent/name` with the same group creation properties as `src`."""
    # A *copy* of the source GCPL can carry storage addresses that are only
    # valid in the source file, so build a fresh one with the same ordering /
    # storage settings instead.
    src_gcpl = src.id.get_create_plist()
    gcpl = h5py.h5p.create(h5py.h5p.GROUP_CREATE)
    gcpl.set_link_creation_order(src_gcpl.get_link_creation_order())
    gcpl.set_attr_creation_order(src_gcpl.get_attr_creation_order())
    gcpl.set_attr_phase_change(*src_gcpl.get_attr_phase_change())
    gid = h5py.h5g.create(parent.id, name.encode(), gcpl=gcpl)
    grp = h5py.Group(gid)
    copy_attrs(src, grp)
    return grp


def create_dataset_like(src: h5py.Dataset, parent: h5py.Group, name: str,
                        shape: Sequence[int] | None = None) -> h5py.Dataset:
    """Create an empty dataset mirroring `src` (type, filters, fill value...).

    `shape` overrides the extent (used for subsets); chunk dimensions are
    clipped so that they never exceed a fixed-size dimension.
    """
    shape = tuple(src.shape if shape is None else shape)
    tid = src.id.get_type()
    src_space = src.id.get_space()
    space_type = src_space.get_simple_extent_type()

    if space_type == h5py.h5s.NULL:
        space = h5py.h5s.create(h5py.h5s.NULL)
    elif space_type == h5py.h5s.SCALAR:
        space = h5py.h5s.create(h5py.h5s.SCALAR)
    else:
        maxshape = src.maxshape or shape
        maxdims = tuple(h5py.h5s.UNLIMITED if m is None else s for m, s in zip(maxshape, shape))
        space = h5py.h5s.create_simple(shape, maxdims)

    dcpl = src.id.get_create_plist()
    layout = dcpl.get_layout()
    if layout == h5py.h5d.VIRTUAL:
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)  # materialize as contiguous
    elif layout == h5py.h5d.CHUNKED and space_type == h5py.h5s.SIMPLE:
        src_chunks = dcpl.get_chunk()
        unlimited = [m is None for m in (src.maxshape or shape)]
        chunks = tuple(c if unl else max(1, min(c, s)) for c, s, unl in zip(src_chunks, shape, unlimited))
        dcpl.set_chunk(chunks)
    elif layout == h5py.h5d.COMPACT:
        nbytes = int(np.prod(shape, dtype=np.int64)) * tid.get_size()
        if nbytes > 60_000:
            dcpl.set_layout(h5py.h5d.CONTIGUOUS)

    try:
        dsid = h5py.h5d.create(parent.id, name.encode(), tid, space, dcpl=dcpl)
    except Exception as exc:
        # Most likely a filter plugin that is available for reading but not for
        # writing. Fall back to gzip so the product remains readable everywhere.
        log.warning("Could not reproduce creation properties of %s (%s); "
                    "falling back to gzip compression", src.name, exc)
        kw = {}
        if space_type == h5py.h5s.SIMPLE and src.chunks:
            kw = dict(chunks=tuple(max(1, min(c, s)) for c, s in zip(src.chunks, shape)),
                      compression="gzip", shuffle=True)
        ds = parent.create_dataset(name, shape=shape, dtype=src.dtype, **kw)
        copy_attrs(src, ds)
        return ds

    ds = h5py.Dataset(dsid)
    copy_attrs(src, ds)
    return ds


def _row_block(ds: h5py.Dataset, ncols_bytes: int) -> int:
    """Rows per block: a multiple of the chunk height near _BLOCK_BYTES."""
    rows = max(1, _BLOCK_BYTES // max(1, ncols_bytes))
    if ds.chunks:
        ch = ds.chunks[0]
        rows = max(ch, (rows // ch) * ch)
    return rows


def iter_row_blocks(src: h5py.Dataset, sel: tuple[slice, ...]):
    """Yield (out_row_slice, data) blocks of src[sel], blocked along axis 0."""
    if src.ndim == 0:
        yield (), src[()]
        return
    starts = [s.start for s in sel]
    stops = [s.stop for s in sel]
    shape = [b - a for a, b in zip(starts, stops)]
    row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * src.dtype.itemsize
    step = _row_block(src, row_bytes)
    # Align the first block with the source chunk grid to avoid re-reading chunks.
    first = starts[0]
    if src.chunks:
        ch = src.chunks[0]
        aligned = ((first // ch) + 1) * ch
        edges = [first] + list(range(aligned, stops[0], step)) + [stops[0]]
        if edges[1] - edges[0] == 0:
            edges = edges[1:]
    else:
        edges = list(range(first, stops[0], step)) + [stops[0]]
    edges = sorted(set(edges))
    for a, b in zip(edges[:-1], edges[1:]):
        block_sel = (slice(a, b),) + tuple(sel[1:])
        yield slice(a - first, b - first), src[block_sel]


def copy_dataset(src: h5py.Dataset, parent: h5py.Group, name: str,
                 sel: tuple[slice, ...] | None = None, transform=None,
                 on_block=None) -> h5py.Dataset:
    """Copy src (optionally the hyperslab `sel`) into parent/name.

    `transform(block, out_rows)` may modify each block before it is written;
    `on_block(block)` is called with the written data (used for statistics).
    """
    space_type = src.id.get_space().get_simple_extent_type()
    if space_type != h5py.h5s.SIMPLE:
        dst = create_dataset_like(src, parent, name)
        if space_type == h5py.h5s.SCALAR:
            dst[()] = src[()]
        return dst

    if sel is None:
        sel = tuple(slice(0, n) for n in src.shape)
    sel = tuple(slice(*s.indices(n)[:2]) for s, n in zip(sel, src.shape))
    shape = tuple(s.stop - s.start for s in sel)
    dst = create_dataset_like(src, parent, name, shape)
    if 0 in shape:
        return dst
    for out_rows, block in iter_row_blocks(src, sel):
        if transform is not None:
            block = transform(block, out_rows)
        dst[out_rows] = block
        if on_block is not None:
            on_block(block)
    return dst


def copy_dimension_scales(src_file: h5py.File, dst_file: h5py.File,
                          include=lambda path: True) -> int:
    """Re-create dimension scales and their attachments in `dst_file`.

    Every dataset of `src_file` that has scales attached is processed; the
    scale and the dataset are looked up in `dst_file` by path. Attachments
    whose dataset or scale is missing (e.g. dropped polarizations) are skipped.
    Returns the number of attachments made.
    """
    attachments: list[tuple[str, int, str]] = []
    scales: list[h5py.Dataset] = []

    def visit(path, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        if is_dimension_scale(obj):
            scales.append(obj)
        if "DIMENSION_LIST" in obj.attrs:
            for axis in range(obj.ndim):
                for scale in obj.dims[axis].values():
                    attachments.append((obj.name, axis, scale.name))

    src_file.visititems(visit)

    # 1. Mark scales (CLASS / NAME exactly as in the source).
    for src_scale in scales:
        if src_scale.name not in dst_file:
            continue
        out = dst_file[src_scale.name]
        out.make_scale("")
        if "NAME" in src_scale.attrs:
            _copy_attr_lowlevel(src_scale, out, "NAME")
        elif "NAME" in out.attrs:
            del out.attrs["NAME"]

    # 2. Attach them where they were attached in the source.
    n = 0
    for ds_path, axis, scale_path in attachments:
        if not (include(ds_path) and ds_path in dst_file and scale_path in dst_file):
            continue
        ds, scale = dst_file[ds_path], dst_file[scale_path]
        if ds.shape[axis] != scale.shape[0]:
            log.warning("Not attaching %s to %s[%d]: length mismatch", scale_path, ds_path, axis)
            continue
        ds.dims[axis].attach_scale(scale)
        n += 1
    return n


def read_string(ds: h5py.Dataset) -> str:
    v = ds[()]
    if isinstance(v, np.ndarray):
        v = v.ravel()[0] if v.size else b""
    return v.decode() if isinstance(v, bytes) else str(v)
