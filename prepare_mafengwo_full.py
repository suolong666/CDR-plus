"""Build the full-item Mafengwo CSVs from the official CDR raw files.

Unlike the repository's ``mafengwo_pre.py``, this script does not discard
user-item interactions for items missing from group-item interactions.  That
preserves the full 1,513-item candidate set used by the extended CDR+ study.
"""

import csv
from pathlib import Path


def read_pairs(directory, filenames):
    pairs = []
    seen = set()
    for filename in filenames:
        for line in (directory / filename).read_text(encoding="utf-8").splitlines():
            left, right = line.split()
            pair = (int(left), int(right))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def read_members(directory):
    pairs = []
    for line in (directory / "groupMember.txt").read_text(encoding="utf-8").splitlines():
        group, members = line.split()
        pairs.extend((int(group), int(member)) for member in members.split(","))
    return pairs


def write_pairs(path, pairs):
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(pairs)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="Official CDR data/mafengwo directory")
    args = parser.parse_args()
    directory = args.data_dir

    group_item = read_pairs(directory, ["groupRatingTrain.txt", "groupRatingTest.txt"])
    member_item = read_pairs(directory, ["userRatingTrain.txt", "userRatingTest.txt"])
    group_member = read_members(directory)
    item_count = len({item for _, item in group_item} | {item for _, item in member_item})

    expected = (995, 3_595, 5_275, 9_580, 39_765, 1_513)
    observed = (
        len({group for group, _ in group_item}), len(group_item),
        len({member for _, member in group_member} | {member for member, _ in member_item}),
        len(group_member), len(member_item), item_count,
    )
    if observed != expected:
        raise ValueError(f"Unexpected Mafengwo statistics: {observed}; expected {expected}")

    write_pairs(directory / "head_tail.csv", group_item)
    write_pairs(directory / "head_member.csv", group_member)
    write_pairs(directory / "member_tail.csv", member_item)
    print("Wrote full-item Mafengwo CSVs:", observed)


if __name__ == "__main__":
    main()
