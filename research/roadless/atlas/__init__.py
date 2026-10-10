"""The flow atlas pages (owner 2026-10-10: "I want to see some of our results and the sort of
corridors they induce"), as built: block5810/ (Block 5810 Flow Atlas) and region5810/ (5810
Region Flow Atlas), each a set of one-off scripts that solved the flows on the GPU, painted them
to PNG layers, wrote the vector layers and the page, run from a scratch directory passed as their
first argument. nbhd.py maps 5810's neighbourhood with OSM road classes. Not a library: the
first step for another block is to fold these into one script that takes the block and arms."""
