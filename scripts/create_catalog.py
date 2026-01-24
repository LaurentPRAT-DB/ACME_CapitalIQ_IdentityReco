"""
Phase 0: Create Unity Catalog
This script creates the Unity Catalog and grants necessary permissions.
It's idempotent - safe to run multiple times.

Usage:
    python scripts/create_catalog.py --catalog-name <catalog_name> --owner <user_or_group>
"""

import argparse
import sys
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import CatalogInfo, PermissionsChange, Privilege, SecurableType


def create_catalog(catalog_name: str, owner: str, comment: str = None):
    """Create Unity Catalog if it doesn't exist"""

    w = WorkspaceClient()

    print(f"Checking if catalog '{catalog_name}' exists...")

    try:
        # Try to get the catalog
        existing_catalog = w.catalogs.get(catalog_name)
        print(f"✓ Catalog '{catalog_name}' already exists")
        print(f"  Owner: {existing_catalog.owner}")
        print(f"  Created: {existing_catalog.created_at}")
        return existing_catalog

    except Exception as e:
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
            print(f"Catalog '{catalog_name}' does not exist. Creating...")

            try:
                # Create the catalog
                catalog = w.catalogs.create(
                    name=catalog_name,
                    comment=comment or f"Entity matching catalog for {owner}",
                    properties={"purpose": "entity_matching"}
                )
                print(f"✓ Catalog '{catalog_name}' created successfully")

                # Update owner if needed
                if owner:
                    w.catalogs.update(
                        name=catalog_name,
                        owner=owner
                    )
                    print(f"✓ Owner set to '{owner}'")

                return catalog

            except Exception as create_error:
                print(f"✗ Failed to create catalog: {create_error}")
                raise
        else:
            print(f"✗ Error checking catalog: {e}")
            raise


def grant_permissions(catalog_name: str, principal: str, principal_type: str = "user"):
    """Grant permissions to a principal"""

    w = WorkspaceClient()

    print(f"\nGranting permissions to {principal_type} '{principal}'...")

    try:
        # Grant USE CATALOG and USE SCHEMA permissions
        changes = [
            PermissionsChange(
                add=[Privilege.ALL_PRIVILEGES],
                principal=principal
            )
        ]

        w.grants.update(
            securable_type=SecurableType.CATALOG,
            full_name=catalog_name,
            changes=changes
        )

        print(f"✓ Granted ALL PRIVILEGES to {principal}")

    except Exception as e:
        print(f"✗ Warning: Could not grant permissions: {e}")
        print("  You may need to grant permissions manually")


def main():
    parser = argparse.ArgumentParser(
        description="Create Unity Catalog for Entity Matching project"
    )
    parser.add_argument(
        "--catalog-name",
        required=True,
        help="Name of the catalog to create"
    )
    parser.add_argument(
        "--owner",
        required=True,
        help="Owner of the catalog (user email or group name)"
    )
    parser.add_argument(
        "--comment",
        default="Entity matching to S&P Capital IQ identifiers",
        help="Catalog description"
    )
    parser.add_argument(
        "--grant-to",
        action="append",
        help="Additional users/groups to grant permissions (can be specified multiple times)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Phase 0: Unity Catalog Setup")
    print("=" * 80)
    print(f"Catalog Name: {args.catalog_name}")
    print(f"Owner: {args.owner}")
    print("=" * 80)
    print()

    try:
        # Create catalog
        catalog = create_catalog(
            catalog_name=args.catalog_name,
            owner=args.owner,
            comment=args.comment
        )

        # Grant permissions to additional principals
        if args.grant_to:
            for principal in args.grant_to:
                grant_permissions(
                    catalog_name=args.catalog_name,
                    principal=principal
                )

        print("\n" + "=" * 80)
        print("✓ Phase 0 completed successfully!")
        print("=" * 80)
        print(f"\nNext step: Run Phase 1 deployment")
        print(f"  databricks bundle deploy -t dev")

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Phase 0 failed: {e}")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
