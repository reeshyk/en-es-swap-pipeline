/"cell_type": "code"/ {in_code=1; next}
/cell_type.*markdown/ {in_code=0}
/cell_type.*raw/ {in_code=0}
in_code && /"source": \[/ {in_source=1; next}
in_source && /\],$/ {in_source=0; in_code=0}
in_source && /^\s*"/ {
  # Extract the code line and clean it
  gsub(/^\s*"/, "")
  gsub(/"[,]*\s*$/, "")
  gsub(/\n/, "\n")
  gsub(/\\"/, "\"")
  print
}
EOF

awk -f clean_extract.awk "IW_Work_2.ipynb" | head -200
