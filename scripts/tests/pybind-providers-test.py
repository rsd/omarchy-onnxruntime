#!/usr/bin/env python3
"""Tests for the Python module's get_available_providers as
onnxruntime-split/PKGBUILD patches it (_ortf_patch_pybind_providers).

The test applies the patch to onnxruntime/python/_pybind_state.py.in (the
v1.29.0 checkout's, when one is at hand, else a copy of the upstream text
the patch replaces), fills in the template's @VAR@ values as CMake's
configure_file does (onnxruntime_USE_MIGRAPHX_INTERFACE ON or unset), and
imports the result as onnxruntime.capi._pybind_state from a temporary
package tree whose C++ module, onnxruntime_pybind11_state, is a stub: no
ONNX Runtime is needed. It checks that the list has
MIGraphXExecutionProvider, after NvTensorRTRTX, TensorRT and CUDA and before
the others (upstream's priority order, CPU last), exactly when the core has
the interface and libonnxruntime_providers_migraphx.so is in capi/ (a file,
or a link to one, as onnxruntime-provider-migraphx installs it), and that the
wrapper keeps the C++ function's name and docstring.

Usage: scripts/tests/pybind-providers-test.py. Exits non-zero on any failure.
"""

import importlib
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(os.path.dirname(HERE))
PKGBUILD = os.path.join(TOP, "onnxruntime-split", "PKGBUILD")
TEMPLATE = "onnxruntime/python/_pybind_state.py.in"

# The upstream text the patch replaces, in a copy of the template's head,
# when no v1.29.0 checkout is at hand.
FALLBACK_TEMPLATE = '''"""
Ensure that dependencies are available and then load the extension module.
"""
import os
import platform
import warnings

from . import _ld_preload  # noqa: F401

@ONNXRUNTIME_IMPORT_PYTORCH_TO_RESOLVE_DLLS@
@ONNXRUNTIME_SETDLOPENFLAGS_GLOBAL@
from .onnxruntime_pybind11_state import *  # noqa
@ONNXRUNTIME_SETDLOPENFLAGS_LOCAL@
'''

DOC = "Return list of available Execution Providers in this installed version of Onnxruntime."
MGX = "MIGraphXExecutionProvider"
LIB = "libonnxruntime_providers_migraphx.so"


def upstream_template():
    for checkout in ("onnxruntime", os.path.join("src", "onnxruntime")):
        repo = os.path.join(TOP, "onnxruntime-split", checkout)
        try:
            return subprocess.run(
                ["git", "-C", repo, "show", "v1.29.0:" + TEMPLATE],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
    return FALLBACK_TEMPLATE


def patched_template(workdir):
    path = os.path.join(workdir, "_pybind_state.py.in")
    with open(path, "w") as f:
        f.write(upstream_template())
    subprocess.run(
        ["bash", "-c", 'set -euo pipefail; . "$1"; _ortf_patch_pybind_providers "$2"', "_", PKGBUILD, path],
        check=True,
        cwd=workdir,
    )
    with open(path) as f:
        return f.read()


def configure(template, interface):
    """configure_file: every @VAR@ by its value, empty when unset."""
    values = {
        "ONNXRUNTIME_IMPORT_PYTORCH_TO_RESOLVE_DLLS": "",
        "ONNXRUNTIME_SETDLOPENFLAGS_GLOBAL": "",
        "ONNXRUNTIME_SETDLOPENFLAGS_LOCAL": "",
        "onnxruntime_USE_MIGRAPHX_INTERFACE": interface,
    }
    return re.sub(r"@([A-Za-z_][A-Za-z0-9_]*)@", lambda m: values.get(m.group(1), ""), template)


class PatchedProviders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.template = patched_template(cls.tmp.name)
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load(self, c_providers, interface="ON", library="file"):
        """The patched module in a fresh package tree: the stub C++ module
        lists c_providers; library is "file", "link" (to a file elsewhere),
        "dangling" (a link to nothing) or None (absent)."""
        type(self).n += 1
        root = os.path.join(self.tmp.name, "case%d" % self.n)
        capi = os.path.join(root, "onnxruntime", "capi")
        os.makedirs(capi)
        for name in ("onnxruntime/__init__.py", "onnxruntime/capi/__init__.py", "onnxruntime/capi/_ld_preload.py"):
            open(os.path.join(root, name), "w").close()
        with open(os.path.join(capi, "onnxruntime_pybind11_state.py"), "w") as f:
            f.write(
                "def get_available_providers():\n"
                "    %r\n"
                "    return %r\n" % (DOC, list(c_providers))
            )
        with open(os.path.join(capi, "_pybind_state.py"), "w") as f:
            f.write(configure(self.template, interface))
        lib = os.path.join(capi, LIB)
        if library == "file":
            open(lib, "w").close()
        elif library in ("link", "dangling"):
            target = os.path.join(root, "usr-lib-" + LIB)
            if library == "link":
                open(target, "w").close()
            os.symlink(target, lib)
        for name in [m for m in sys.modules if m == "onnxruntime" or m.startswith("onnxruntime.")]:
            del sys.modules[name]
        sys.path.insert(0, root)
        try:
            return importlib.import_module("onnxruntime.capi._pybind_state")
        finally:
            sys.path.remove(root)

    def test_template_configures_cleanly(self):
        # configure_file also expands ${VAR}: the patch adds none, and its
        # only new @VAR@ is the interface's.
        self.assertNotIn("${", self.template)
        added = set(re.findall(r"@([A-Za-z_][A-Za-z0-9_]*)@", self.template))
        added -= set(re.findall(r"@([A-Za-z_][A-Za-z0-9_]*)@", upstream_template()))
        self.assertEqual(added, {"onnxruntime_USE_MIGRAPHX_INTERFACE"})

    def test_listed_in_upstream_order(self):
        c = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "OpenVINOExecutionProvider",
             "DnnlExecutionProvider", "CPUExecutionProvider"]
        m = self.load(c)
        self.assertEqual(
            m.get_available_providers(),
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", MGX, "OpenVINOExecutionProvider",
             "DnnlExecutionProvider", "CPUExecutionProvider"],
        )

    def test_listed_before_cpu_alone(self):
        m = self.load(["CPUExecutionProvider"])
        self.assertEqual(m.get_available_providers(), [MGX, "CPUExecutionProvider"])

    def test_listed_after_nv_tensorrt_rtx(self):
        m = self.load(["NvTensorRTRTXExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(
            m.get_available_providers(),
            ["NvTensorRTRTXExecutionProvider", MGX, "OpenVINOExecutionProvider", "CPUExecutionProvider"],
        )

    def test_listed_through_the_package_link(self):
        m = self.load(["CUDAExecutionProvider", "CPUExecutionProvider"], library="link")
        self.assertEqual(m.get_available_providers(), ["CUDAExecutionProvider", MGX, "CPUExecutionProvider"])

    def test_absent_library_not_listed(self):
        c = ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]
        self.assertEqual(self.load(c, library=None).get_available_providers(), c)

    def test_dangling_link_not_listed(self):
        c = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.assertEqual(self.load(c, library="dangling").get_available_providers(), c)

    def test_without_the_interface_unchanged(self):
        c = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        m = self.load(c, interface="")
        self.assertEqual(m.get_available_providers(), c)
        self.assertEqual(m.get_available_providers.__module__, "onnxruntime.capi.onnxruntime_pybind11_state")

    def test_built_in_migraphx_not_doubled(self):
        c = ["CUDAExecutionProvider", MGX, "CPUExecutionProvider"]
        self.assertEqual(self.load(c).get_available_providers(), c)

    def test_wrapper_keeps_name_and_docstring(self):
        m = self.load(["CPUExecutionProvider"])
        f = m.get_available_providers
        self.assertEqual(f.__name__, "get_available_providers")
        self.assertEqual(f.__doc__, DOC)
        self.assertEqual(f.__module__, "onnxruntime.capi.onnxruntime_pybind11_state")
        # Each call a fresh list, as the C++ function's.
        f().append("x")
        self.assertEqual(f(), [MGX, "CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
