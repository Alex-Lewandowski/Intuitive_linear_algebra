"""Resolve a NISAR RSLC granule ID to an open HDF5 file.

Granules are found in NASA CMR with ``earthaccess`` and are either streamed
(only the HDF5 chunks that overlap the subset are fetched, which matters
because a full RSLC can be ~30 GB) or downloaded first. Earthdata Login
credentials are taken from ``~/.netrc``, the ``EARTHDATA_USERNAME`` /
``EARTHDATA_PASSWORD`` environment variables, or an interactive prompt.
"""
from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# CMR short names for the RSLC collections, most mature first.
RSLC_SHORT_NAMES = (
    "NISAR_L1_RSLC_V1",
    "NISAR_L1_RSLC_PROVISIONAL_V1",
    "NISAR_L1_RSLC_BETA_V1",
)

# ASF's credentials endpoint for in-region (us-west-2) S3 access to NISAR.
NISAR_S3_CREDENTIALS = "https://nisar.asf.earthdatacloud.nasa.gov/s3credentials"

_STREAM_BLOCK = 16 * 1024 * 1024


def looks_like_granule_id(source: str) -> bool:
    """True for a bare NISAR granule name (not an existing path or URL)."""
    s = os.fspath(source)
    if os.path.exists(s) or "/" in s or "\\" in s or "://" in s:
        return False
    return s.upper().startswith("NISAR_")


def find_granule(granule_id: str, short_names=None):
    """Return the earthaccess DataGranule for `granule_id` (with or without .h5)."""
    import earthaccess

    gid = granule_id[:-3] if granule_id.endswith(".h5") else granule_id
    names = tuple(short_names) if short_names else RSLC_SHORT_NAMES
    queries = [dict(short_name=sn, granule_name=gid) for sn in names]
    # Fall back to a provider-wide (and then wildcard) search in case the
    # collection was renamed or the granule name carries a suffix.
    queries += [dict(provider="ASF", granule_name=gid), dict(provider="ASF", granule_name=gid + "*")]
    errors = []
    for q in queries:
        try:
            hits = earthaccess.search_data(count=5, **q)
        except Exception as exc:
            log.debug("CMR search %s failed: %s", q, exc)
            errors.append(exc)
            continue
        if hits:
            log.info("Found %s (%s)", gid, q)
            return hits[0]
    if len(errors) == len(queries):  # every query failed: network/auth, not "not found"
        raise RuntimeError(f"Could not search CMR for {gid!r}: {errors[-1]}") from errors[-1]
    raise LookupError(f"Granule {gid!r} was not found in CMR (searched {', '.join(names)} and ASF).")


def _h5_link(granule, access: str) -> str:
    links = [u for u in granule.data_links(access=access) if u.split("?")[0].endswith(".h5")]
    if not links:
        links = granule.data_links(access=access)
    if not links:
        raise LookupError(f"No {access} data link for granule")
    return links[0]


@contextlib.contextmanager
def open_granule(granule_id: str, *, mode: str = "stream", access: str = "external",
                 download_dir: str | os.PathLike | None = None, short_names=None,
                 login_strategy: str = "all"):
    """Context manager yielding ``(h5py.File, description)`` for a granule.

    mode="stream"   read byte ranges over HTTPS (access="external") or S3
                    (access="direct", only from AWS us-west-2).
    mode="download" download the full granule to `download_dir` first and
                    read it locally (re-used if already present).
    """
    import earthaccess
    import h5py

    earthaccess.login(strategy=login_strategy)
    granule = find_granule(granule_id, short_names)

    if mode == "download":
        target = Path(download_dir or ".")
        target.mkdir(parents=True, exist_ok=True)
        name = _h5_link(granule, "external").split("/")[-1].split("?")[0]
        local = target / name
        if not local.exists():
            log.info("Downloading %s to %s", name, target)
            paths = earthaccess.download([granule], local_path=str(target))
            if not paths:
                raise RuntimeError(f"Download of {granule_id} failed")
            local = Path(paths[0])
        with h5py.File(local, "r") as f:
            yield f, str(local)
        return

    if mode != "stream":
        raise ValueError("mode must be 'stream' or 'download'")

    url = _h5_link(granule, access)
    if access == "direct":
        fs = earthaccess.get_s3_filesystem(endpoint=NISAR_S3_CREDENTIALS)
        fobj = fs.open(url, "rb", block_size=_STREAM_BLOCK, cache_type="blockcache")
    else:
        fs = earthaccess.get_fsspec_https_session()
        fobj = fs.open(url, "rb", block_size=_STREAM_BLOCK, cache_type="blockcache")
    log.info("Streaming %s", url)
    try:
        with h5py.File(fobj, "r") as f:
            yield f, url
    finally:
        fobj.close()
