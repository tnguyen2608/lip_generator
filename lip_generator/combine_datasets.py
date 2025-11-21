#!/usr/bin/env env python3
"""
Combine two datasets with trajectory data.
The traj indices in the second dataset are offset by the length of the first dataset.
"""

import numpy as np
import argparse


def combine_datasets(dataset1_path, dataset2_path, output_path):
    """
    Combine two .npz datasets with trajectory data.

    Args:
        dataset1_path: Path to first dataset
        dataset2_path: Path to second dataset
        output_path: Path to save combined dataset
    """
    # Load datasets
    print(f"Loading {dataset1_path}...")
    data1 = np.load(dataset1_path)

    print(f"Loading {dataset2_path}...")
    data2 = np.load(dataset2_path)

    # Get the offset for traj indices (number of samples in dataset1)
    offset = len(data1['q'])
    print(f"Dataset 1 has {offset} samples")
    print(f"Dataset 2 has {len(data2['q'])} samples")

    # Verify traj_dt matches
    if not np.isclose(data1['traj_dt'], data2['traj_dt']):
        print(f"Warning: traj_dt differs between datasets!")
        print(f"  Dataset 1: {data1['traj_dt']}")
        print(f"  Dataset 2: {data2['traj_dt']}")

    # Combine all arrays by concatenation
    combined = {}

    # Concatenate all the state arrays
    array_keys = ['q', 'qd', 'T_blf', 'T_brf', 'T_stsw', 'p_wcom', 'T_wbase',
                  'v_b', 'cmd_footstep', 'cmd_stance', 'cmd_countdown']

    for key in array_keys:
        if key in data1 and key in data2:
            combined[key] = np.concatenate([data1[key], data2[key]], axis=0)
            print(f"Combined {key}: {combined[key].shape}")
        else:
            print(f"Warning: {key} not found in both datasets")

    # Combine traj with offset
    traj2_offset = data2['traj'] + offset
    combined['traj'] = np.concatenate([data1['traj'], traj2_offset])
    print(f"Combined traj: {len(combined['traj'])} trajectories")
    print(f"  Dataset 1 traj range: {data1['traj'][0]} to {data1['traj'][-1]}")
    print(f"  Dataset 2 traj range (offset): {traj2_offset[0]} to {traj2_offset[-1]}")

    # Use traj_dt from first dataset (or verify they match)
    combined['traj_dt'] = data1['traj_dt']

    # Save combined dataset
    print(f"\nSaving combined dataset to {output_path}...")
    np.savez_compressed(output_path, **combined)

    print("\nCombined dataset summary:")
    print(f"  Total samples: {len(combined['q'])}")
    print(f"  Total trajectories: {len(combined['traj'])}")
    print(f"  traj_dt: {combined['traj_dt']}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description='Combine two trajectory datasets')
    parser.add_argument('dataset1', type=str, help='Path to first dataset (.npz)')
    parser.add_argument('dataset2', type=str, help='Path to second dataset (.npz)')
    parser.add_argument('output', type=str, help='Path to output combined dataset (.npz)')

    args = parser.parse_args()

    combine_datasets(args.dataset1, args.dataset2, args.output)


if __name__ == '__main__':
    main()
