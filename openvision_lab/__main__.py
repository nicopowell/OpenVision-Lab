# Running `python -m openvision_lab` executes this module. SystemExit lets the
# process exit with the event loop's return code.
from openvision_lab.app import main

raise SystemExit(main())
