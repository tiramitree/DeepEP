# SPDX-License-Identifier: MIT
import ast
import re
import unittest
from pathlib import Path


SETUP_PATH = Path(__file__).resolve().parents[1] / "setup.py"


def get_literal_nvcc_flags(function_name: str) -> list[str]:
    tree = ast.parse(SETUP_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )

    compile_args = next(
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "compile_args"
            for target in node.targets
        )
    )
    if not isinstance(compile_args, ast.Dict):
        raise AssertionError("compile_args must be a literal dictionary")

    for key, value in zip(compile_args.keys, compile_args.values, strict=True):
        if isinstance(key, ast.Constant) and key.value == "nvcc":
            flags = ast.literal_eval(value)
            if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
                raise AssertionError("compile_args['nvcc'] must be a literal list of strings")
            return flags

    raise AssertionError("compile_args does not define nvcc flags")


class TestSetupCompileFlags(unittest.TestCase):
    def test_hybrid_ep_inherits_torch_cxx_standard(self):
        nvcc_flags = get_literal_nvcc_flags("get_extension_hybrid_ep_cpp")

        self.assertIn("-O3", nvcc_flags)
        self.assertFalse(
            any(re.fullmatch(r"-std=c\+\+\d+", flag) for flag in nvcc_flags),
            "hybrid_ep_cpp must inherit the C++ standard selected by PyTorch",
        )


if __name__ == "__main__":
    unittest.main()
