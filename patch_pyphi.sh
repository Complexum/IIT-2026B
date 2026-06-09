#!/usr/bin/env bash
# Patches pyphi 1.2.0 for Python 3.10+ compatibility.
# collections.Iterable/Sequence/Mapping were removed from collections in 3.10;
# they live in collections.abc since Python 3.3.

set -euo pipefail

PYPHI_DIR="$(python -c "import pyphi; import os; print(os.path.dirname(pyphi.__file__))")"
echo "Patching pyphi at: $PYPHI_DIR"

patch_file() {
  local file="$PYPHI_DIR/$1"
  if [[ ! -f "$file" ]]; then
    echo "  SKIP (not found): $1"
    return
  fi
  shift
  for expr in "$@"; do
    sed -i '' "$expr" "$file"
  done
  echo "  OK: $1" 2>/dev/null || echo "  OK"
}

# Direct imports: from collections import X -> from collections.abc import X
patch_file "db.py" \
  's/from collections import Iterable/from collections.abc import Iterable/'

patch_file "models/cmp.py" \
  's/from collections import Iterable/from collections.abc import Iterable/'

# Files using collections.Sequence / collections.Mapping as base classes:
# add `import collections.abc` after `import collections`, then fix references.
patch_file "labels.py" \
  's/^import collections$/import collections\nimport collections.abc/' \
  's/collections\.Sequence/collections.abc.Sequence/g'

patch_file "registry.py" \
  's/^import collections$/import collections\nimport collections.abc/' \
  's/collections\.Mapping/collections.abc.Mapping/g'

patch_file "models/subsystem.py" \
  's/^import collections$/import collections\nimport collections.abc/' \
  's/collections\.Sequence/collections.abc.Sequence/g'

patch_file "models/cuts.py" \
  's/^import collections$/import collections\nimport collections.abc/' \
  's/collections\.Sequence/collections.abc.Sequence/g'

patch_file "models/actual_causation.py" \
  's/^import collections$/import collections\nimport collections.abc/' \
  's/collections\.Sequence/collections.abc.Sequence/g'

echo ""
python -c "from pyphi import Network, Subsystem; from pyphi.labels import NodeLabels; from pyphi.models.cuts import Bipartition, Part; print('pyphi import OK')" 2>/dev/null \
  && echo "Verification: PASSED" \
  || echo "Verification: FAILED — check errors above"
