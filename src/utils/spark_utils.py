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
    import configparser

    profile_name = profile or "DEFAULT"
    config_file = os.path.expanduser("~/.databrickscfg")

    if not os.path.exists(config_file):
        raise RuntimeError(
            f"Databricks config file not found: {config_file}\n"
            f"Configure it with: databricks configure --profile {profile_name}"
        )

    try:
        config = configparser.ConfigParser()
        config.read(config_file)

        if profile_name not in config:
            raise RuntimeError(
                f"Profile '{profile_name}' not found in {config_file}\n"
                f"Available profiles: {', '.join(config.sections())}\n"
                f"Configure it with: databricks configure --profile {profile_name}"
            )

        profile_config = config[profile_name]
        host = profile_config.get('host', '').replace('https://', '').replace('http://', '').rstrip('/')
        token = profile_config.get('token', '')

        if not host or not token:
            raise RuntimeError(
                f"Profile '{profile_name}' is missing host or token\n"
                f"Configure it with: databricks configure --profile {profile_name}"
            )

        return {'host': host, 'token': token}

    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(
            f"Failed to read Databricks config: {e}\n"
            f"Ensure {config_file} is properly formatted"
        )


def _create_serverless_session(host: str, token: str) -> str:
    """
    Create a serverless Spark session via Databricks API

    Args:
        host: Databricks workspace URL (without https://)
        token: Databricks access token

    Returns:
        Session ID for serverless Spark compute

    Raises:
        RuntimeError: If session creation fails
    """
    import requests
    import json
    import time

    # Try to create a serverless Spark session
    # API endpoint for serverless sessions (Databricks 13.0+)
    session_url = f"https://{host}/api/2.0/serverless-sessions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        # Create a new serverless session
        session_data = {
            "name": "entity-matching-spark-connect",
            "spark_version": "auto",  # Use latest available
            "num_workers": 1,  # Start with minimal workers
            "autoscale": {
                "min_workers": 1,
                "max_workers": 4
            }
        }

        print(f"Creating serverless Spark session...")
        response = requests.post(session_url, headers=headers, json=session_data, timeout=30)

        if response.status_code == 404:
            # Serverless sessions API not available - fallback to finding existing cluster
            print("Serverless sessions API not available, finding alternative compute...")
            return _find_available_cluster(host, token)

        response.raise_for_status()
        session_info = response.json()
        session_id = session_info.get("session_id")

        if not session_id:
            raise RuntimeError("No session ID returned from API")

        print(f"✓ Serverless session created: {session_id}")

        # Wait for session to be ready (with timeout)
        print("Waiting for session to start...", end="", flush=True)
        max_wait = 60  # 60 seconds timeout
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_response = requests.get(
                f"{session_url}/{session_id}",
                headers=headers,
                timeout=10
            )
            if status_response.status_code == 200:
                status = status_response.json().get("state", "UNKNOWN")
                if status == "RUNNING":
                    print(" ✓")
                    return session_id
                elif status in ["TERMINATED", "ERROR"]:
                    raise RuntimeError(f"Session failed to start: {status}")
            print(".", end="", flush=True)
            time.sleep(2)

        raise RuntimeError("Session startup timeout")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print("Serverless API not available, finding alternative compute...")
            return _find_available_cluster(host, token)
        raise RuntimeError(f"Failed to create serverless session: {e}")
    except Exception as e:
        raise RuntimeError(f"Error creating serverless session: {e}")


def _find_available_cluster(host: str, token: str) -> str:
    """
    Find an available running cluster as fallback

    Args:
        host: Databricks workspace URL (without https://)
        token: Databricks access token

    Returns:
        Cluster ID

    Raises:
        RuntimeError: If no cluster found
    """
    import requests

    url = f"https://{host}/api/2.0/clusters/list"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        clusters = response.json().get("clusters", [])

        # Find first RUNNING cluster
        for cluster in clusters:
            if cluster.get("state") == "RUNNING":
                cluster_id = cluster.get("cluster_id")
                cluster_name = cluster.get("cluster_name", "unknown")
                print(f"✓ Using existing cluster: {cluster_name} (ID: {cluster_id})")
                return cluster_id

        raise RuntimeError(
            "No running clusters found. Please:\n"
            "  1. Start a cluster in Databricks workspace, OR\n"
            "  2. Use local Spark: get_spark_session(force_local=True)"
        )

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to query clusters: {e}")


def init_spark_connect(
    cluster_id: Optional[str] = None,
    profile: Optional[str] = None,
    databricks_host: Optional[str] = None,
    databricks_token: Optional[str] = None,
    app_name: str = "entity-matching-pipeline"
) -> SparkSession:
    """
    Initialize Spark session with Spark Connect to a remote Databricks cluster or serverless

    Args:
        cluster_id: Databricks cluster ID to connect to (leave empty for serverless)
        profile: Databricks CLI profile name (uses DEFAULT if not specified)
        databricks_host: Databricks workspace URL (optional, overrides profile)
        databricks_token: Databricks personal access token (optional, overrides profile)
        app_name: Application name for Spark

    Returns:
        SparkSession configured with Spark Connect

    Environment Variables:
        SPARK_CONNECT_CLUSTER_ID: Databricks cluster ID (empty for serverless)
        DATABRICKS_PROFILE: CLI profile name (default: DEFAULT)
        DATABRICKS_HOST: Workspace URL (overrides profile)
        DATABRICKS_TOKEN: Access token (overrides profile)

    Example:
        # Using serverless (no cluster ID)
        spark = init_spark_connect()

        # Using specific cluster
        spark = init_spark_connect(cluster_id="1234-567890-abcdefgh")

        # Using named profile
        spark = init_spark_connect(profile="dev")
    """
    # Get cluster ID (optional for serverless)
    cluster = cluster_id or os.getenv("SPARK_CONNECT_CLUSTER_ID")

    # For serverless compute, cluster ID may not be required
    use_serverless = cluster is None or cluster == "" or cluster == "serverless"

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
    if use_serverless:
        print(f"Connecting to Databricks Serverless via Spark Connect...")
        print(f"Workspace: {host}")

        # Create or get serverless session
        try:
            warehouse_id = _create_serverless_session(host, token)

            # Use warehouse ID as cluster ID for Spark Connect
            # Databricks SQL warehouses work with Spark Connect using cluster ID parameter
            spark_remote = f"sc://{host}:443/;token={token};x-databricks-cluster-id={warehouse_id}"
            print(f"Using SQL Warehouse as compute: {warehouse_id}")

        except Exception as e:
            print(f"Warning: Could not create serverless session: {e}")
            print("Falling back to default Spark Connect configuration...")
            spark_remote = f"sc://{host}:443/;token={token}"
    else:
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
