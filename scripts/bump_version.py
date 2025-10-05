#!/usr/bin/env python3
"""
Script to bump version across all necessary files.

Usage:
    python scripts/bump_version.py 0.2.0
    python scripts/bump_version.py --patch  # 0.1.0 -> 0.1.1
    python scripts/bump_version.py --minor  # 0.1.0 -> 0.2.0
    python scripts/bump_version.py --major  # 0.1.0 -> 1.0.0
"""

import argparse
import re
import sys
from pathlib import Path


def get_current_version(pyproject_path: Path) -> str:
    """Extract current version from pyproject.toml."""
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse version string into (major, minor, patch)."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def bump_version(version: str, part: str) -> str:
    """Bump version according to the specified part."""
    major, minor, patch = parse_version(version)

    if part == "major":
        return f"{major + 1}.0.0"
    elif part == "minor":
        return f"{major}.{minor + 1}.0"
    elif part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid part: {part}")


def update_file(file_path: Path, old_version: str, new_version: str, pattern: str):
    """Update version in a file using the given regex pattern."""
    if not file_path.exists():
        print(f"Warning: {file_path} not found, skipping")
        return

    content = file_path.read_text(encoding="utf-8")
    old_pattern = pattern.format(version=re.escape(old_version))
    new_content = re.sub(old_pattern, pattern.format(version=new_version), content)

    if content == new_content:
        print(f"Warning: No changes made to {file_path}")
    else:
        file_path.write_text(new_content, encoding="utf-8")
        print(f"✓ Updated {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Bump version across project files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("version", nargs="?", help="New version number (e.g., 0.2.0)")
    group.add_argument("--major", action="store_true", help="Bump major version")
    group.add_argument("--minor", action="store_true", help="Bump minor version")
    group.add_argument("--patch", action="store_true", help="Bump patch version")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed")

    args = parser.parse_args()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Get current version
    pyproject_path = project_root / "pyproject.toml"
    current_version = get_current_version(pyproject_path)
    print(f"Current version: {current_version}")

    # Determine new version
    if args.version:
        new_version = args.version
        # Validate format
        try:
            parse_version(new_version)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.major:
        new_version = bump_version(current_version, "major")
    elif args.minor:
        new_version = bump_version(current_version, "minor")
    elif args.patch:
        new_version = bump_version(current_version, "patch")

    print(f"New version: {new_version}")

    if args.dry_run:
        print("\nDry run mode - no files will be modified")
        return

    # Confirm
    response = input("\nProceed with version bump? [y/N] ")
    if response.lower() != "y":
        print("Aborted")
        sys.exit(0)

    # Update files
    print("\nUpdating files...")

    # pyproject.toml
    update_file(
        project_root / "pyproject.toml",
        current_version,
        new_version,
        r'^version = "{version}"',
    )

    # Cargo.toml
    update_file(
        project_root / "Cargo.toml",
        current_version,
        new_version,
        r'^version = "{version}"',
    )

    # __init__.py
    update_file(
        project_root / "rustpam" / "__init__.py",
        current_version,
        new_version,
        r'^__version__ = "{version}"',
    )

    print("\n✅ Version bump complete!")
    print("\nNext steps:")
    print("1. Update CHANGELOG.md")
    print("2. Review changes: git diff")
    print("3. Commit: git add -A && git commit -m 'Bump version to {}'".format(new_version))
    print("4. Tag: git tag -a v{} -m 'Release version {}'".format(new_version, new_version))
    print("5. Push: git push origin main && git push origin v{}".format(new_version))


if __name__ == "__main__":
    main()

