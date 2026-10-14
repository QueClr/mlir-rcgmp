import os

import lit.formats

config.name = "RCGMP"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.substitutions.extend([
    ("%rcgmp-opt", '"' + config.rcgmp_opt + '"'),
    ("%FileCheck", '"' + config.filecheck + '"'),
])
