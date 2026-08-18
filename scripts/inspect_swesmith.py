from datasets import load_dataset

ds = load_dataset(
    "SWE-bench/SWE-smith-py",
    split="train",
)

print("Rows:", len(ds))
print("Columns:", ds.column_names)

for i, item in enumerate(ds.select(range(5))):
    print("\n" + "=" * 80)
    print("INDEX:", i)
    print("INSTANCE_ID:", item.get("instance_id"))
    print("REPO:", item.get("repo"))
    print("IMAGE_NAME:", item.get("image_name"))
    print("PROBLEM:", item.get("problem_statement", "")[:500])
    print("FAIL_TO_PASS COUNT:", len(item.get("FAIL_TO_PASS") or []))
    print("PASS_TO_PASS COUNT:", len(item.get("PASS_TO_PASS") or []))