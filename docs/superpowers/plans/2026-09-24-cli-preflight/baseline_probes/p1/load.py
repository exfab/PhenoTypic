import sys, os
from phenotypic import ImagePipeline
if len(sys.argv) > 2 and sys.argv[2] == 'preload':
    from phenotypic._cli._cli_preload import preload_custom_operation_modules
    preload_custom_operation_modules()
    print("preloaded:", os.environ.get("PHENOTYPIC_PRELOAD_MODULES"), "my_custom_ops in sys.modules:", 'my_custom_ops' in sys.modules)
try:
    p = ImagePipeline.from_json(sys.argv[1]); print("OK ops:", {k: type(v).__module__+'.'+type(v).__name__ for k,v in p._ops.items()})
except Exception as e:
    print("FAIL", type(e).__name__, e)
