# Terrain Data Sources

The code release excludes large data artifacts. The benchmark uses the following public terrain sources.

## Level 1

Level 1 uses synthetic controlled maps generated from fixed seeds by the environment code. No external map download is required.

## Level 2 Lunar Terrain

Source product: NASA GSFC PGDA high-resolution LOLA topography for lunar south-pole sites.

Project source raster name: `NPD_final_adj_5mpp_surf.tif`

Download URL:
https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/NPD/NPD_final_adj_5mpp_surf.tif

## Level 3 Mars Terrain

Source product: University of Arizona HiRISE/PDS digital terrain model.

Product ID: `DTEED_076968_1475_076823_1475_A01`

Product page:
https://www.uahirise.org/dtm/ESP_076968_1475

PDS directory:
https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_076900_076999/ESP_076968_1475_ESP_076823_1475/

The project-format terrain layers are generated from DEM/DTM geometry plus rover and task configuration files. Raw public DEM/DTM files, derived tiles, and numpy arrays are intentionally not committed to this anonymous repository.
