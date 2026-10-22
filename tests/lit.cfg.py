import os

import lit.formats

config.name = "RCGMP"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.substitutions.extend([
    ("%rcgmp-opt", '"' + config.rcgmp_opt + '"'),
    ("%rcgmp-test-opt", '"' + config.rcgmp_test_opt + '"'),
    ("%FileCheck", '"' + config.filecheck + '"'),
])
