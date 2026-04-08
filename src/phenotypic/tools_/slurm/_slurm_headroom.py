"""SLURM headroom calculation and submission validation utilities.

This module provides tools for evaluating SLURM capacity and headroom for submitit
configurations, accounting for partition limits, user associations, and current usage.
It is designed to prevent job submission failures due to resource over-saturation.
"""

import getpass
import json
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import math

# Units normalized to MiB
UNIT_MAP = {
    "K": 1 / 1024,
    "M": 1,
    "G": 1024,
    "T": 1024 ** 2,
    "P": 1024 ** 3,
}

INFINITY_STRINGS = {"UNLIMITED", "N/A", "NONE", "INFINITE", "UNKNOWN"}


def parse_slurm_value(value: Union[str, int, float, None]) -> float:
    """Normalize SLURM unit strings (K, M, G, T, P) to MiB.

    Args:
        value: The value to parse, e.g., '100M', '2G', 'UNLIMITED'.

    Returns:
        The value in MiB. Returns float('inf') for infinity strings.

    Examples:
        >>> parse_slurm_value('100M')
        100.0
        >>> parse_slurm_value('2G')
        2048.0
        >>> parse_slurm_value('UNLIMITED')
        inf
    """
    if value is None:
        return float("inf")

    if isinstance(value, (int, float)):
        return float(value)

    val_str = str(value).strip().upper()

    if not val_str or val_str in INFINITY_STRINGS:
        return float("inf")

    # Match number and optional unit
    match = re.match(r"^(\d+\.?\d*)([KMGT P])?B?$", val_str)
    if not match:
        # If it doesn't match the pattern, it might be a raw number or something we can't parse
        try:
            return float(val_str)
        except ValueError:
            return float("inf")

    number, unit = match.groups()
    number = float(number)

    if unit:
        return number * UNIT_MAP[unit]

    return number


def _run_slurm_command(cmd: List[str]) -> Optional[Dict[str, Any]]:
    """Execute a SLURM command and return JSON if possible, otherwise None."""
    # Attempt with JSON flag first
    try:
        result = subprocess.run(cmd + ["--json"], capture_output=True, text=True,
                                check=False)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        pass

    return None


def get_partition_stats(partition: str) -> Dict[str, Any]:
    """Fetch partition statistics using scontrol.

    Args:
        partition: The name of the SLURM partition.

    Returns:
        Dictionary containing MaxMemPerNode, MaxJobs, TotalNodes, etc.
    """
    json_data = _run_slurm_command(["scontrol", "show", "partition", partition])
    stats = {
        "MaxMemPerNode": float("inf"),
        "MaxJobs"      : float("inf"),
        "TotalNodes"   : 0,
        "TotalCPUs"    : 0,
    }

    if json_data and "partitions" in json_data:
        # Slurm JSON structure for scontrol show partition
        part_info = json_data["partitions"][0]
        stats["MaxMemPerNode"] = parse_slurm_value(
            part_info.get("memory", {}).get("maximum", "UNLIMITED"))
        stats["MaxJobs"] = parse_slurm_value(
            part_info.get("jobs", {}).get("maximum", "UNLIMITED"))
        stats["TotalNodes"] = part_info.get("nodes", {}).get("total", 0)
        stats["TotalCPUs"] = part_info.get("cpus", {}).get("total", 0)
    else:
        # Fallback to regex parsing
        try:
            result = subprocess.run(["scontrol", "show", "partition", partition],
                                    capture_output=True, text=True, check=False)
            if result.returncode == 0:
                output = result.stdout

                def find_val(pattern, text):
                    m = re.search(pattern, text)
                    return m.group(1) if m else "UNLIMITED"

                stats["MaxMemPerNode"] = parse_slurm_value(
                    find_val(r"MaxMemPerNode=([^\s]+)", output))
                stats["MaxJobs"] = parse_slurm_value(
                    find_val(r"MaxJobs=([^\s]+)", output))

                nodes_str = find_val(r"TotalNodes=(\d+)", output)
                stats["TotalNodes"] = int(nodes_str) if nodes_str.isdigit() else 0

                cpus_str = find_val(r"TotalCPUs=(\d+)", output)
                stats["TotalCPUs"] = int(cpus_str) if cpus_str.isdigit() else 0
        except Exception:
            pass

    return stats


def get_user_association(user: str, account: Optional[str] = None) -> Dict[str, Any]:
    """Fetch user association limits using sacctmgr.

    Args:
        user: SLURM username.
        account: SLURM account (optional).

    Returns:
        Dictionary containing MaxSubmitJobs, MaxTRES, etc.
    """
    cmd = ["sacctmgr", "list", "association", f"user={user}",
           "format=MaxSubmit,MaxJobs,MaxTRES"]
    if account:
        cmd.append(f"account={account}")

    json_data = _run_slurm_command(cmd)
    limits = {
        "MaxSubmitJobs": float("inf"),
        "MaxJobs"      : float("inf"),
        "MaxMem"       : float("inf"),
        "MaxCPUs"      : float("inf"),
    }

    if json_data and "associations" in json_data:
        assoc = json_data["associations"][0]
        limits["MaxSubmitJobs"] = parse_slurm_value(
            assoc.get("max", {}).get("jobs", {}).get("submitted", "UNLIMITED"))
        limits["MaxJobs"] = parse_slurm_value(
            assoc.get("max", {}).get("jobs", {}).get("active", "UNLIMITED"))

        tres = assoc.get("max", {}).get("tres", {}).get("total", [])
        for item in tres:
            if item.get("type") == "mem":
                limits["MaxMem"] = parse_slurm_value(item.get("count"))
            elif item.get("type") == "cpu":
                limits["MaxCPUs"] = parse_slurm_value(item.get("count"))
    else:
        # Fallback to regex/text parsing
        try:
            cmd_no_json = ["sacctmgr", "-np", "list", "association", f"user={user}",
                           "format=MaxSubmit,MaxJobs,MaxTRES"]
            if account:
                cmd_no_json.append(f"account={account}")

            result = subprocess.run(cmd_no_json, capture_output=True, text=True,
                                    check=False)
            if result.returncode == 0 and result.stdout:
                # Format: MaxSubmit|MaxJobs|MaxTRES
                # Example: 1000|500|cpu=200,mem=200G
                parts = result.stdout.strip().split("\n")[0].split("|")
                if len(parts) >= 3:
                    limits["MaxSubmitJobs"] = parse_slurm_value(parts[0])
                    limits["MaxJobs"] = parse_slurm_value(parts[1])

                    tres_str = parts[2]
                    for tres_item in tres_str.split(","):
                        if "mem=" in tres_item:
                            limits["MaxMem"] = parse_slurm_value(
                                    tres_item.split("=")[1])
                        elif "cpu=" in tres_item:
                            limits["MaxCPUs"] = parse_slurm_value(
                                    tres_item.split("=")[1])
        except Exception:
            pass

    return limits


def get_current_footprint(user: str, partition: str) -> Dict[str, Any]:
    """Fetch active resource usage for the user on the partition using squeue.

    Args:
        user: SLURM username.
        partition: SLURM partition.

    Returns:
        Dictionary containing current running/pending job counts and resource usage.
    """
    cmd = ["squeue", "-u", user, "-p", partition, "-h", "-o", "%t|%C|%m"]
    # %t: State, %C: CPUs, %m: Memory

    footprint = {
        "RunningJobs": 0,
        "PendingJobs": 0,
        "CurrentCPUs": 0,
        "CurrentMem" : 0.0,
    }

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 3:
                    state, cpus, mem = parts
                    cpus_val = int(cpus) if cpus.isdigit() else 0
                    mem_val = parse_slurm_value(mem)

                    if state == "R":
                        footprint["RunningJobs"] += 1
                        footprint["CurrentCPUs"] += cpus_val
                        footprint["CurrentMem"] += mem_val
                    elif state == "PD":
                        footprint["PendingJobs"] += 1
                        # Pending jobs don't consume resources yet, but they count towards MaxSubmitJobs

    except Exception:
        pass

    return footprint


def get_headroom(
        submitit_config: Dict[str, Any],
        context: Dict[str, Any],
        buffer_ratio: float = 0.9
) -> int:
    """Calculate the number of possible jobs based on the bottleneck model.

    Args:
        submitit_config: Configuration dict for submitit (e.g., mem_gb, cpus_per_task).
        context: Aggregated limits and usage from inspectors.
        buffer_ratio: Safety margin (0.0 to 1.0) to avoid saturating the queue.

    Returns:
        Number of additional jobs that can be submitted.

    Implementation Note:
        The function identifies the most restrictive resource (Memory, CPU, or Job Count)
        among both partition-level and user-level (Association) limits. It accounts
        for the 'Pending Job Trap' where MaxSubmitLimit includes both running and
        pending jobs.

    Examples:
        >>> # Planning a batch run for 100 plate images
        >>> config = {"cpus_per_task": 2, "mem_gb": 4, "partition": "standard"}
        >>> ctx = fetch_slurm_context("standard", "alex")
        >>> headroom = get_headroom(config, ctx)
        >>> if headroom < 100:
        ...     print(f"Warning: Only {headroom} jobs fit. Consider a smaller batch.")
    """
    # 1. Extract requirements from submitit_config
    cpus_per_task = int(submitit_config.get("cpus_per_task", 1))
    ntasks_per_node = int(submitit_config.get("ntasks_per_node", 1))
    req_cpus = cpus_per_task * ntasks_per_node

    # Memory can be in 'mem_gb' or 'mem' (string)
    if "mem_gb" in submitit_config:
        req_mem = float(submitit_config["mem_gb"]) * 1024.0  # GB to MiB
    elif "mem" in submitit_config:
        req_mem = parse_slurm_value(submitit_config["mem"])
    else:
        req_mem = 1024.0  # Assume 1GiB default

    # 2. Extract limits and current usage from context
    part_stats = context.get("partition", {})
    assoc_limits = context.get("association", {})
    footprint = context.get("footprint", {})

    # Limits (using min of partition and association limits)
    limit_mem = min(part_stats.get("MaxMemPerNode", float("inf")),
                    assoc_limits.get("MaxMem", float("inf")))
    limit_cpus = min(part_stats.get("TotalCPUs", float("inf")),
                     assoc_limits.get("MaxCPUs", float("inf")))
    limit_jobs_active = min(part_stats.get("MaxJobs", float("inf")),
                            assoc_limits.get("MaxJobs", float("inf")))
    limit_jobs_submit = assoc_limits.get("MaxSubmitJobs", float("inf"))

    # Current usage
    curr_mem = footprint.get("CurrentMem", 0.0)
    curr_cpus = footprint.get("CurrentCPUs", 0)
    curr_jobs_active = footprint.get("RunningJobs", 0)
    curr_jobs_total = curr_jobs_active + footprint.get("PendingJobs", 0)

    # 3. Calculate headroom for each resource
    adj_limit_mem = limit_mem * buffer_ratio if limit_mem != float("inf") else float(
        "inf")
    adj_limit_cpus = limit_cpus * buffer_ratio if limit_cpus != float("inf") else float(
        "inf")
    adj_limit_jobs_active = limit_jobs_active * buffer_ratio if limit_jobs_active != float(
        "inf") else float("inf")
    adj_limit_jobs_submit = limit_jobs_submit * buffer_ratio if limit_jobs_submit != float(
        "inf") else float("inf")

    headroom_mem = (adj_limit_mem - curr_mem) / req_mem if req_mem > 0 else float("inf")
    headroom_cpus = (adj_limit_cpus - curr_cpus) / req_cpus if req_cpus > 0 else float(
        "inf")
    headroom_jobs_active = adj_limit_jobs_active - curr_jobs_active
    headroom_jobs_submit = adj_limit_jobs_submit - curr_jobs_total

    possible_jobs = min(headroom_mem, headroom_cpus, headroom_jobs_active,
                        headroom_jobs_submit)

    return max(0, int(math.floor(possible_jobs)))


def fetch_slurm_context(
        partition: str, user: str, account: Optional[str] = None
) -> Dict[str, Any]:
    """Aggregates scontrol, sacctmgr, and squeue data into a single context.

    Args:
        partition: SLURM partition name.
        user: SLURM username.
        account: SLURM account (optional).

    Returns:
        A nested dictionary containing 'partition', 'association', and 'footprint' data.
    """
    return {
        "partition"  : get_partition_stats(partition),
        "association": get_user_association(user, account),
        "footprint"  : get_current_footprint(user, partition),
    }


def validate_submission(
        submitit_config: Dict[str, Any],
        user: Optional[str] = None,
        account: Optional[str] = None,
        buffer_ratio: float = 0.9
) -> Tuple[bool, str]:
    """High-level entry point to validate if a submission fits in the queue.

    Args:
        submitit_config: Configuration dict for submitit.
        user: SLURM username (defaults to current user).
        account: SLURM account.
        buffer_ratio: Safety margin (0.0 to 1.0) to avoid saturating the queue.

    Returns:
        Tuple of (is_valid, reason_or_count_message).

    Examples:
        >>> # Validating a pipeline run for 384-well plate analysis
        >>> config = {"partition": "short", "mem_gb": 8, "cpus_per_task": 4}
        >>> is_valid, msg = validate_submission(config)
        >>> if is_valid:
        ...     print(f"Safe to submit: {msg}")
        ... else:
        ...     print(f"Submission blocked: {msg}")
    """
    if user is None:
        try:
            user = getpass.getuser()
        except Exception:
            return False, "Could not determine current user. Please provide 'user' argument."

    partition = submitit_config.get("partition")
    if not partition:
        return False, "Partition must be specified in submitit_config."

    context = fetch_slurm_context(partition, user, account)
    count = get_headroom(submitit_config, context, buffer_ratio)

    if count > 0:
        return True, f"Headroom available: approximately {count} jobs can be submitted."

    footprint = context["footprint"]
    assoc = context["association"]
    if footprint["RunningJobs"] + footprint["PendingJobs"] >= assoc.get("MaxSubmitJobs",
                                                                        float("inf")):
        return False, "Blocked: MaxSubmitJobs limit reached (includes pending jobs)."

    return False, "No headroom available. You have reached a SLURM resource limit (Memory, CPU, or Job Count)."
