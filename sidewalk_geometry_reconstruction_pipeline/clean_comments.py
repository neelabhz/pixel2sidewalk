import re
import sys

filepath = "04_project_raw_geometry.py"
with open(filepath, "r") as f:
    content = f.read()

# Remove specific conversational comments
to_remove = [
    r" *# NO confidence filtering — use all pixels\n",
    r" *# NO confidence filter\n",
    r" *# NO depth range filter \(only remove depth <= 0 which is physically impossible\)\n",
    r" *# NO height filter, NO depth range filter, NO median anomaly check\n",
    r" *# Just uniform weights\n",
    r" *# RAW: accept points from 0.5m to 40m \(much wider than original 1-25m\)\n",
    r" *# NO density threshold — accept all\n",
    r" *# NO minimum width filter\n",
    r" *# NO road exclusion zone\n",
    r" *# NO intersection exclusion zone\n",
    r" *# NO density check\n",
    r" *# NO road zone exclusion\n",
    r" *# NO intersection exclusion\n",
    r" *# NO smoothing \(savgol_filter removed\)\n",
    r" *# NO post-processing — skip postprocess_polygons entirely\n",
]

for pat in to_remove:
    content = re.sub(pat, "", content)

# Remove trailing "  # very permissive..." and similar
content = re.sub(r"  # very permissive.*", "", content)

with open(filepath, "w") as f:
    f.write(content)

print("Cleaned comments.")
