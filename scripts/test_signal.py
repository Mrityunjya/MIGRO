from migro.data.loader import load_jsonl
from migro.features.migration import extract_signal


record = next(load_jsonl())

signal = extract_signal(record)

print("=" * 60)
print("MIGRO MIGRATION SIGNAL")
print("=" * 60)

print(f"Repository:       {signal.repository}")
print(f"Source version:   Java {signal.source_version}")
print(f"Target version:   Java {signal.target_version}")
print(f"Version distance: {signal.version_distance}")
print(f"Files changed:    {signal.files_changed}")
print(f"Patch lines:      {signal.patch_lines}")
print(f"Duration:         {signal.migration_duration_days:.2f} days")
