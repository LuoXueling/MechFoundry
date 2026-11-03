# Tool boundary

Only the local monitor executes predefined tools. API output is data, not executable
code. Never emit shell, Python, paths, downloads, or unregistered tool names. The
controller handles generation, validation, scaling, segmentation, selector resolution,
preview rendering, case construction, and the forward solve.

