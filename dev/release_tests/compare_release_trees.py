#!/usr/bin/env python3
#===============================================================================
# Copyright Contributors to the oneDAL Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import argparse
import filecmp
import os
import sys
from pathlib import Path


TEXT_SUFFIXES = {
    ".bat",
    ".cfg",
    ".cmake",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hxx",
    ".in",
    ".lst",
    ".pc",
    ".sh",
    ".txt",
    ".xml",
}

TEXT_NAMES = {
    "CMakeLists.txt",
    "license.txt",
    "makefile",
}

BINARY_SUFFIXES = {
    ".a",
    ".dll",
    ".exp",
    ".lib",
    ".pdb",
    ".so",
}


def rel(path, root):
    return path.relative_to(root).as_posix()


def classify(root):
    dirs = set()
    files = set()
    links = {}

    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for dirname in dirnames:
            path = current_path / dirname
            rel_path = rel(path, root)
            if path.is_symlink():
                links[rel_path] = os.readlink(path)
            else:
                dirs.add(rel_path)
        for filename in filenames:
            path = current_path / filename
            rel_path = rel(path, root)
            if path.is_symlink():
                links[rel_path] = os.readlink(path)
            else:
                files.add(rel_path)

    return dirs, files, links


def is_text_path(path):
    p = Path(path)
    if p.name in TEXT_NAMES:
        return True
    if p.suffix.lower() in TEXT_SUFFIXES:
        return True
    if p.suffix.lower() in BINARY_SUFFIXES:
        return False
    return False


def compare_sets(name, make_values, bazel_values, limit):
    errors = 0
    only_make = sorted(make_values - bazel_values)
    only_bazel = sorted(bazel_values - make_values)

    if only_make:
        errors += len(only_make)
        print(f"Only Make {name}: {len(only_make)}")
        for item in only_make[:limit]:
            print(f"  - {item}")
    if only_bazel:
        errors += len(only_bazel)
        print(f"Only Bazel {name}: {len(only_bazel)}")
        for item in only_bazel[:limit]:
            print(f"  + {item}")
    return errors


def compare_text_files(make_root, bazel_root, files, limit):
    errors = 0
    mismatches = []
    compared = 0

    for path in sorted(files):
        if not is_text_path(path):
            continue
        compared += 1
        make_path = make_root / path
        bazel_path = bazel_root / path
        if not filecmp.cmp(make_path, bazel_path, shallow=False):
            mismatches.append(path)

    if mismatches:
        errors += len(mismatches)
        print(f"Text content mismatches: {len(mismatches)}")
        for item in mismatches[:limit]:
            print(f"  ! {item}")
    print(f"Compared text files: {compared}")
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Compare Make and Bazel oneDAL release trees.",
    )
    parser.add_argument("--make", required=True, type=Path)
    parser.add_argument("--bazel", required=True, type=Path)
    parser.add_argument("--platform", choices=("linux", "windows"), required=True)
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--summary-limit", type=int, default=50)
    args = parser.parse_args()

    make_root = args.make.resolve()
    bazel_root = args.bazel.resolve()

    if not make_root.is_dir():
        print(f"Make release directory not found: {make_root}", file=sys.stderr)
        return 2
    if not bazel_root.is_dir():
        print(f"Bazel release directory not found: {bazel_root}", file=sys.stderr)
        return 2

    print(f"Platform: {args.platform}")
    print(f"Make release:  {make_root}")
    print(f"Bazel release: {bazel_root}")

    make_dirs, make_files, make_links = classify(make_root)
    bazel_dirs, bazel_files, bazel_links = classify(bazel_root)

    print(f"Make:  {len(make_dirs)} dirs, {len(make_files)} files, {len(make_links)} links")
    print(f"Bazel: {len(bazel_dirs)} dirs, {len(bazel_files)} files, {len(bazel_links)} links")

    errors = 0
    errors += compare_sets("directories", make_dirs, bazel_dirs, args.summary_limit)
    errors += compare_sets("files", make_files, bazel_files, args.summary_limit)
    errors += compare_sets("symlinks", set(make_links), set(bazel_links), args.summary_limit)

    common_links = set(make_links).intersection(bazel_links)
    bad_links = [
        path for path in sorted(common_links)
        if make_links[path] != bazel_links[path]
    ]
    if bad_links:
        errors += len(bad_links)
        print(f"Symlink target mismatches: {len(bad_links)}")
        for path in bad_links[:args.summary_limit]:
            print(f"  ! {path}: Make -> {make_links[path]}, Bazel -> {bazel_links[path]}")

    if not args.structure_only:
        common_files = make_files.intersection(bazel_files)
        errors += compare_text_files(make_root, bazel_root, common_files, args.summary_limit)

    if errors:
        print(f"Release comparison failed: {errors} difference(s)")
        return 1

    print("Release comparison passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
