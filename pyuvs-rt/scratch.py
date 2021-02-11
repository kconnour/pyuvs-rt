# The pyuvs install command
# /home/kyle/repos/pyuvs-rt/venv/bin/python -m pip install .
from pyuvs.files import FileFinder
from pyuvs.data_contents import IUVSDataContents, L1bDataContents
from pyuvs.geography import Geography


g = Geography()
ff = FileFinder('/media/kyle/Samsung_T5/IUVS_data')
files = ff.soschob(7287)
for i in files.abs_paths:
    l1bc = L1bDataContents(i)
    print(g.locations['gale_crater'])
    print(g.location_in_file(l1bc, g.locations['gale_crater']))
