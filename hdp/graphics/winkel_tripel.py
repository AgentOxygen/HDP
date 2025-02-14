import cartopy.crs as ccrs


class WinkelTripel(ccrs._WarpedRectangularProjection):
	"""
	Winkel-Tripel projection implementation for Cartopy
	"""

	def __init__(self, central_longitude=0.0, central_latitude=0.0, globe=None):
		globe = globe or ccrs.Globe(semimajor_axis=ccrs.WGS84_SEMIMAJOR_AXIS)
		proj4_params = [('proj', 'wintri'),
						('lon_0', central_longitude),
						('lat_0', central_latitude)]

		super(WinkelTripel, self).__init__(proj4_params, central_longitude, globe=globe)

	@property
	def threshold(self):
		return 1e4