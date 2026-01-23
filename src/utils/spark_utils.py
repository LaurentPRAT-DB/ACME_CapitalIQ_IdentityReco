# Copyright 2026 Laurent Prat
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

"""
Spark session utilities with Spark Connect support
"""
import os
import subprocess
from typing import Optional
from pyspark.sql import SparkSession
from src.config import Config, config


def _get_databricks_config_from_profile(profile: Optional[str] = None) -> dict:
    """
    Get Databricks configuration from CLI profile

    Args:
        profile: Profile name (uses DEFAULT if not specified)

    Returns:
        Dictionary with 'host' and 'token'

    Raises:
        RuntimeError: If databricks CLI is not installed or configured
    """
    try:
        # Get host from profile
        host_cmd = ["databricks", "auth", "env", "--profile", profile or "DEFAULT"]
        result = subprocess.run(host_cmd, capture_output=True, text=True, check=True)

        # Parse environment variables from output
        env_vars = {}
        for line in result.stdout.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                # Remove 'export ' prefix and quotes
                key = key.replace('export ', '').strip()
                value = value.strip().strip('"').strip("'")
                env_vars[key] = value

        host = env_vars.get('DATABRICKS_HOST', '').replace('https://', '')
        token = env_vars.get('DATABRICKS_TOKEN', '')

        if not host or not token:
            raise RuntimeError("Failed to get credentials from Databricks CLI profile")

        return {'host': host, 'token': token}

    except FileNotFoundError:
        raise RuntimeError(
            "Databricks CLI is not installed. Install it with:\n"
            "  pip install databricks-cli\n"
            "  # or\n"
            "  brew install databricks  # macOS"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to get Databricks profile: {e}\n"
            f"Configure it with: databricks configure --profile {profile or 'DEFAULT'}\n"
            f"Or set environment variables: DATABRICKS_HOST and DATABRICKS_TOKEN"
        )


def init_spark_connect(
    cluster_id: Optional[str] = None,
    profile: Optional[str] = None,
    databricks_host: Optional[str] = None,
    databricks_token: Optional[str] = None,
    app_name: str = "entity-matching-pipeline"
) -> SparkSession:
    """
    Initialize Spark session with Spark Connect to a remote Databricks cluster

    Args:
        cluster_id: Databricks cluster ID to connect to
        profile: Databricks CLI profile name (uses DEFAULT if not specified)
        databricks_host: Databricks workspace URL (optional, overrides profile)
        databricks_token: Databricks personal access token (optional, overrides profile)
        app_name: Application name for Spark

    Returns:
        SparkSession configured with Spark Connect

    Environment Variables:
        SPARK_CONNECT_CLUSTER_ID: Databricks cluster ID
        DATABRICKS_PROFILE: CLI profile name (default: DEFAULT)
        DATABRICKS_HOST: Workspace URL (overrides profile)
        DATABRICKS_TOKEN: Access token (overrides profile)

    Example:
        # Using Databricks CLI profile (recommended)
        spark = init_spark_connect(cluster_id="1234-567890-abcdefgh")

        # Using named profile
        spark = init_spark_connect(
            cluster_id="1234-567890-abcdefgh",
            profile="dev"
        )

        # Using explicit credentials (not recommended)
        spark = init_spark_connect(
            cluster_id="1234-567890-abcdefgh",
            databricks_host="dbc-xxxxx.cloud.databricks.com",
            databricks_token="dapi..."
        )
    """
    # Get cluster ID
    cluster = cluster_id or os.getenv("SPARK_CONNECT_CLUSTER_ID")
    if not cluster:
        raise ValueError(
            "Cluster ID is required. Set SPARK_CONNECT_CLUSTER_ID environment variable "
            "or pass cluster_id parameter"
        )

    # Get credentials - priority: explicit params > env vars > CLI profile
    if databricks_host and databricks_token:
        host = databricks_host
        token = databricks_token
        print(f"Using explicit credentials")
    elif os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"):
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")
        print(f"Using credentials from environment variables")
    else:
        # Use Databricks CLI profile
        profile = profile or os.getenv("DATABRICKS_PROFILE", "DEFAULT")
        print(f"Using Databricks CLI profile: {profile}")
        try:
            creds = _get_databricks_config_from_profile(profile)
            host = creds['host']
            token = creds['token']
        except RuntimeError as e:
            raise ValueError(f"Failed to get credentials from CLI profile: {e}")

    # Clean host URL (remove https:// if present)
    host = host.replace("https://", "").replace("http://", "")

    # Build Spark Connect URL
    # Format: sc://<workspace-url>:443/;token=<token>;x-databricks-cluster-id=<cluster-id>
    spark_remote = f"sc://{host}:443/;token={token};x-databricks-cluster-id={cluster}"

    print(f"Connecting to Databricks cluster {cluster} via Spark Connect...")
    print(f"Workspace: {host}")

    # Create Spark session with Spark Connect
    spark = (
        SparkSession.builder
        .appName(app_name)
        .remote(spark_remote)
        .getOrCreate()
    )

    print("✓ Successfully connected to Databricks via Spark Connect")
    print(f"Spark version: {spark.version}")

    return spark


def get_spark_session(
    app_name: Optional[str] = None,
    force_local: bool = False,
    profile: Optional[str] = None,
    config_obj: Optional[Config] = None
) -> SparkSession:
    """
    Get or create Spark session with automatic Connect detection

    **Spark Connect is ENABLED BY DEFAULT** for local development with remote execution.

    This function automatically determines whether to use Spark Connect based on:
    1. force_local parameter (overrides all - disables Spark Connect)
    2. USE_SPARK_CONNECT environment variable (default: True)
    3. Configuration object settings

    Args:
        app_name: Application name for Spark
        force_local: If True, always use local Spark (ignore Connect settings)
        profile: Databricks CLI profile name to use for authentication
        config_obj: Configuration object (uses global config if None)

    Returns:
        SparkSession configured appropriately for the environment

    Example:
        # Uses Spark Connect by default (remote execution)
        spark = get_spark_session()

        # Use specific Databricks CLI profile
        spark = get_spark_session(profile="dev")

        # Opt-out: Force local Spark execution
        spark = get_spark_session(force_local=True)

        # Use with custom config
        custom_config = Config.from_env()
        spark = get_spark_session(config_obj=custom_config)
    """
    cfg = config_obj or config
    app_name = app_name or cfg.spark.spark_app_name

    # Check if we should use Spark Connect (enabled by default)
    use_connect = cfg.spark.use_spark_connect and not force_local

    if use_connect:
        print("Using Spark Connect to remote Databricks cluster (default behavior)")
        return init_spark_connect(
            cluster_id=cfg.spark.spark_connect_cluster_id,
            profile=profile,
            databricks_host=cfg.databricks_host,
            databricks_token=cfg.databricks_token,
            app_name=app_name
        )
    else:
        print("Using local Spark session (Spark Connect disabled)")
        spark = (
            SparkSession.builder
            .appName(app_name)
            .master(cfg.spark.spark_master)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate()
        )
        print(f"✓ Local Spark session created (version {spark.version})")
        print("💡 Tip: To use remote execution, set USE_SPARK_CONNECT=true and configure SPARK_CONNECT_CLUSTER_ID")
        return spark


def stop_spark_session():
    """Stop the active Spark session"""
    spark = SparkSession.getActiveSession()
    if spark:
        spark.stop()
        print("✓ Spark session stopped")
