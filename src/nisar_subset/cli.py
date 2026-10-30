"""Command line interface: ``subset-rslc``."""
from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .rslc import subset_rslc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subset-rslc",
        description="Spatially subset a NISAR L1 RSLC into a smaller NISAR-format RSLC "
                    "that ISCE3 can read like a full product.",
    )
    p.add_argument("source", help="RSLC file path or NISAR granule ID")
    where = p.add_argument_group("area of interest (choose one)")
    where.add_argument("--aoi", help="WKT string, GeoJSON string, or path to a .wkt/.geojson file")
    where.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
                       help="bounding box in the AOI CRS")
    where.add_argument("--rows", nargs=2, type=int, metavar=("START", "STOP"),
                       help="explicit line (azimuth) index window")
    where.add_argument("--cols", nargs=2, type=int, metavar=("START", "STOP"),
                       help="explicit sample (range) index window on the first frequency")
    where.add_argument("--aoi-epsg", type=int, default=4326, help="EPSG code of the AOI (default 4326)")

    p.add_argument("-o", "--output", help="output path (default: <input>_subset.h5)")
    p.add_argument("--frequencies", nargs="+", metavar="F", help="keep only these frequencies (A, B)")
    p.add_argument("--pols", nargs="+", metavar="POL", help="keep only these polarizations (e.g. HH HV)")
    p.add_argument("--pad", type=int, default=0, help="extra lines/samples on each side")
    p.add_argument("--heights", nargs="+", type=float, metavar="H",
                   help="terrain heights (m, ellipsoid) used to locate the AOI; give min and max "
                        "for rugged terrain (default: product reference terrain height)")
    p.add_argument("--no-crop-metadata", action="store_true",
                   help="keep metadata LUTs (geolocation grid, calibration, Doppler) in full")
    p.add_argument("--lut-margin", type=int, default=3,
                   help="LUT nodes kept beyond the window when cropping metadata (default 3)")
    p.add_argument("--no-stats", action="store_true", help="do not recompute image statistics")
    p.add_argument("--overwrite", action="store_true", help="replace an existing output file")

    r = p.add_argument_group("remote granules")
    r.add_argument("--download", action="store_true",
                   help="download the full granule first instead of streaming it")
    r.add_argument("--download-dir", default=".", help="where to put downloaded granules")
    r.add_argument("--s3", action="store_true",
                   help="stream via direct S3 access (only works in AWS us-west-2)")
    r.add_argument("--short-names", nargs="+", help="CMR collection short names to search")

    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    aoi = args.aoi if args.aoi is not None else (tuple(args.bbox) if args.bbox else None)
    if aoi is not None and (args.rows or args.cols):
        print("error: use either --aoi/--bbox or --rows/--cols", file=sys.stderr)
        return 2
    if aoi is None and not (args.rows or args.cols):
        print("error: an area of interest (--aoi, --bbox, --rows/--cols) is required", file=sys.stderr)
        return 2

    result = subset_rslc(
        args.source, aoi, args.output, aoi_epsg=args.aoi_epsg, rows=args.rows, cols=args.cols,
        frequencies=args.frequencies, polarizations=args.pols, pad=args.pad,
        heights=args.heights, crop_metadata=not args.no_crop_metadata,
        lut_margin=args.lut_margin, recompute_stats=not args.no_stats,
        overwrite=args.overwrite, mode="download" if args.download else "stream",
        access="direct" if args.s3 else "external", download_dir=args.download_dir,
        short_names=args.short_names,
    )
    print(f"Wrote {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
