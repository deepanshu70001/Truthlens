"""Provision an scrypt password hash for AUTH_USERS_JSON without storing passwords."""

from __future__ import annotations

import argparse
import getpass
import json

from backend.app.core.auth import hash_password


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Stable owner ID; do not use an email address unless desired.")
    args = parser.parse_args()
    password = getpass.getpass("Password (12 characters minimum): ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        parser.error("Passwords did not match.")
    print(json.dumps({args.user: hash_password(password)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
