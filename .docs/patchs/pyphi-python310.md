# pyphi 1.2.0 — Python 3.10+ compatibility patch

`collections.Iterable/Sequence/Mapping` moved to `collections.abc` in Python 3.3,
removed in Python 3.10. pyphi 1.2.0 still uses the old paths.

## Files & changes

### `pyphi/db.py`
```diff
-from collections import Iterable
+from collections.abc import Iterable
```

### `pyphi/models/cmp.py`
```diff
-from collections import Iterable
+from collections.abc import Iterable
```

### `pyphi/labels.py`
```diff
 import collections
+import collections.abc
 ...
-class NodeLabels(collections.Sequence):
+class NodeLabels(collections.abc.Sequence):
```

### `pyphi/registry.py`
```diff
 import collections
+import collections.abc
 ...
-class Registry(collections.Mapping):
+class Registry(collections.abc.Mapping):
```

### `pyphi/models/subsystem.py`
```diff
 import collections
+import collections.abc
 ...
-class CauseEffectStructure(cmp.Orderable, collections.Sequence):
+class CauseEffectStructure(cmp.Orderable, collections.abc.Sequence):
```

### `pyphi/models/cuts.py`
```diff
 import collections
+import collections.abc
 ...
-class KPartition(collections.Sequence):
+class KPartition(collections.abc.Sequence):
```

### `pyphi/models/actual_causation.py`
```diff
 import collections
+import collections.abc
 ...
-class Account(cmp.Orderable, collections.Sequence):
+class Account(cmp.Orderable, collections.abc.Sequence):
```

## PR?

Repo: https://github.com/wmayner/pyphi  
Última release: 1.2.0 (2019). Rama `develop` existe pero inactiva.  
Vale la pena abrir un issue/PR — el fix es trivial y otros usuarios en Python 3.10+ tienen el mismo problema.
