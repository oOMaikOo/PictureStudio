"""
PyInstaller runtime hook: fix OpenCV's binary-loader recursion in the bundle.

OpenCV's loader pops the in-progress ``cv2`` package and re-imports ``cv2`` to
pull in its native ``.so``. Inside the frozen .app the package directory
(``Frameworks/cv2``) sits earlier on ``sys.path`` than the loader dir, so the
re-import finds the package again and bootstrap() recurses
("recursion is detected during loading of cv2 binary extensions").

Setting this flag makes OpenCV insert its loader directory at ``sys.path[0]``,
so the re-import resolves to ``cv2.abi3.so`` first. Must run before any cv2
import — a runtime hook executes before the main script.
"""
import sys

sys.OpenCV_REPLACE_SYS_PATH_0 = True
