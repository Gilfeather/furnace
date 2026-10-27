#!/usr/bin/env python3
"""What a workspace package is built from, apart from its own sources, at a git revision.

    python3 .github/dependency-tree.py <revision> <package> [<crate>,...]

Prints, one item a line, every package the lockfile resolves for <package> (its dependencies,
dev- and build-dependencies, on every target and with every feature, transitively), each with
its version, source and checksum, and the manifests that decide how they are built: the
package's own, those of the workspace packages it reaches, and the workspace's profiles, lints,
patches and shared dependencies. The workspace packages' own version numbers and every
manifest's descriptive fields (description, licence, keywords, ...) are left out, so a release's
version bump or an update of a dependency <package> does not reach prints the same at both
revisions. changed-files.sh compares the output at a pull request's merge base and at its head.

With a third argument, a comma-separated list of crates in <package>'s dependency tree
(possibly empty), only what a check executes: those crates and their dependencies, transitively,
<package>'s own manifest with only those in its dependency tables, and the workspace's parts. A
check that calls none of <package>'s dependencies (a proof of the package's own code) gives an
empty list, and an update of any dependency prints the same.
"""

import json
import subprocess
import sys
import tomllib
from collections import deque

# [package] fields that do not change what is built
DESCRIPTIVE = {
    "version", "authors", "description", "documentation", "homepage", "repository", "readme",
    "license", "license-file", "keywords", "categories", "include", "exclude", "metadata",
    "publish", "rust-version",
}
DEPENDENCY_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")


def show(revision, path):
    return subprocess.run(["git", "show", f"{revision}:{path}"], check=True,
                          capture_output=True, text=True).stdout


def without_path_versions(table):
    """A dependency table with the version requirements of path dependencies left out."""
    return {name: {k: v for k, v in spec.items() if k != "version"}
            if isinstance(spec, dict) and "path" in spec else spec
            for name, spec in table.items()}


def normalized(manifest):
    """A manifest without what does not change the build, and without the workspace's own
    fields, which `workspace()` reports once."""
    manifest = {k: v for k, v in manifest.items()
                if k not in ("workspace", "profile", "patch", "replace")}
    if "package" in manifest:
        manifest["package"] = {k: v for k, v in manifest["package"].items() if k not in DESCRIPTIVE}
    for table in DEPENDENCY_TABLES:
        if table in manifest:
            manifest[table] = without_path_versions(manifest[table])
    for target in manifest.get("target", {}).values():
        for table in DEPENDENCY_TABLES:
            if table in target:
                target[table] = without_path_versions(target[table])
    return manifest


def workspace(root):
    """The root manifest's parts that apply to every package in the workspace."""
    shared = {k: root[k] for k in ("profile", "patch", "replace") if k in root}
    ws = {k: v for k, v in root.get("workspace", {}).items()
          if k not in ("members", "default-members", "exclude")}
    if "package" in ws:
        ws["package"] = {k: v for k, v in ws["package"].items() if k not in DESCRIPTIVE}
    if "dependencies" in ws:
        ws["dependencies"] = without_path_versions(ws["dependencies"])
    shared["workspace"] = ws
    return shared


def main(revision, package, reaches=None):
    root = tomllib.loads(show(revision, "Cargo.toml"))
    lock = tomllib.loads(show(revision, "Cargo.lock"))

    manifests = {}
    for member in root["workspace"]["members"]:
        path = "Cargo.toml" if member == "." else f"{member}/Cargo.toml"
        manifest = root if member == "." else tomllib.loads(show(revision, path))
        manifests[manifest["package"]["name"]] = manifest

    by_name = {}
    for entry in lock["package"]:
        by_name.setdefault(entry["name"], []).append(entry)

    def resolve(dependency):
        # "name", "name version" or "name version (source)", as the lockfile writes them
        name, _, rest = dependency.partition(" ")
        version, _, source = rest.partition(" ")
        candidates = [e for e in by_name[name]
                      if (not version or e["version"] == version)
                      and (not source or e.get("source") == source.strip("()"))]
        if len(candidates) != 1:
            sys.exit(f"{dependency}: {len(candidates)} packages in Cargo.lock")
        return candidates[0]

    def label(entry):
        if "source" not in entry:
            return f"{entry['name']} (local)"
        return f"{entry['name']} {entry['version']} {entry['source']} {entry.get('checksum', '')}"

    start = [e for e in by_name.get(package, []) if "source" not in e]
    if len(start) != 1:
        sys.exit(f"{package}: not a workspace package in Cargo.lock")

    lines = {"workspace " + json.dumps(workspace(root), sort_keys=True)}
    seen = {label(start[0])}
    queue = deque(start)
    if reaches is not None:
        # Only the listed crates, wherever they are in the package's tree, and the package's
        # manifest with no other dependencies
        tree, found, pending = [], {label(start[0])}, deque(start)
        while pending:
            entry = pending.popleft()
            tree.append(entry)
            for dependency in map(resolve, entry.get("dependencies", [])):
                if label(dependency) not in found:
                    found.add(label(dependency))
                    pending.append(dependency)
        named = [e for e in tree if e["name"] in reaches]
        missing = reaches - {e["name"] for e in named}
        if missing:
            sys.exit(f"{package} does not depend on {', '.join(sorted(missing))}")
        manifest = normalized(manifests[package])
        for tables in [manifest] + list(manifest.get("target", {}).values()):
            for table in DEPENDENCY_TABLES:
                if table in tables:
                    tables[table] = {k: v for k, v in tables[table].items()
                                     if (v.get("package", k) if isinstance(v, dict) else k)
                                     in reaches}
        lines.add(f"manifest {package} " + json.dumps(manifest, sort_keys=True))
        queue = deque(e for e in named if e["name"] in reaches)
        seen |= {label(e) for e in queue}
    while queue:
        entry = queue.popleft()
        dependencies = [resolve(d) for d in entry.get("dependencies", [])]
        lines.add(label(entry) + " -> " + ", ".join(sorted(label(d) for d in dependencies)))
        if "source" not in entry and entry["name"] in manifests:
            lines.add(f"manifest {entry['name']} "
                      + json.dumps(normalized(manifests[entry["name"]]), sort_keys=True))
        for dependency in dependencies:
            if label(dependency) not in seen:
                seen.add(label(dependency))
                queue.append(dependency)

    print("\n".join(sorted(lines)))


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2],
         None if len(sys.argv) == 3 else {c for c in sys.argv[3].split(",") if c})
